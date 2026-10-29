"""Le réseau TC-ResNet8 et trois réseaux de comparaison (Rubin, Noman, Deng), tous sur la même entrée MFCC.

TC-ResNet8 est un réseau de convolutions 1D pensé pour l'audio embarqué. Les 13 MFCC sont les canaux
d'entrée et les filtres ne glissent que dans le temps, ce qui garde le réseau petit.

Entrée : (lot, 13, 75). Sortie : un score par fenêtre (plus il est haut, plus c'est anormal).
Les canaux passent de 10 à 15, 20 puis 30 (25 056 paramètres).
"""
import torch
import torch.nn as nn

CANAUX = (10, 15, 20, 30)


class Bloc(nn.Module):
    """Bloc résiduel : deux convolutions plus un raccourci qui additionne l'entrée à la sortie."""

    def __init__(self, c_in, c_out, pas=2):
        super().__init__()
        # chemin principal ; pas=2 sur la première convolution divise la durée par 2
        self.conv = nn.Sequential(
            nn.Conv1d(c_in, c_out, 9, pas, 4, bias=False), nn.BatchNorm1d(c_out), nn.ReLU(),
            nn.Conv1d(c_out, c_out, 9, 1, 4, bias=False), nn.BatchNorm1d(c_out))
        # le raccourci (convolution de taille 1) remet l'entrée à la même forme que le chemin principal
        self.raccourci = nn.Sequential(nn.Conv1d(c_in, c_out, 1, pas, bias=False), nn.BatchNorm1d(c_out))

    def forward(self, x):
        return torch.relu(self.conv(x) + self.raccourci(x))


class TCResNet8(nn.Module):
    """Une convolution d'entrée, trois blocs résiduels qui divisent la durée par 2 (75 -> 38 -> 19 -> 10 instants),
    moyenne sur le temps, puis une couche linéaire qui donne le score de la fenêtre."""

    def __init__(self):
        super().__init__()
        c = CANAUX
        # première convolution : 13 MFCC -> 10 canaux
        self.entree = nn.Sequential(nn.Conv1d(13, c[0], 3, 1, 1, bias=False), nn.BatchNorm1d(c[0]), nn.ReLU())
        # trois blocs : 10 -> 15 -> 20 -> 30 canaux
        self.blocs = nn.Sequential(Bloc(c[0], c[1]), Bloc(c[1], c[2]), Bloc(c[2], c[3]))
        # couche finale : 30 valeurs -> 1 score
        self.sortie = nn.Linear(c[3], 1)

    def caracteristiques(self, x):
        # moyenne sur le temps : une valeur par canal
        return self.blocs(self.entree(x)).mean(dim=2)

    def forward(self, x):
        return self.sortie(self.caracteristiques(x)).squeeze(1)


class Rubin(nn.Module):
    """CNN 2D sur la carte MFCC (13 x 75 vue comme une image), dans l'esprit de Rubin et al. (2017) :
    deux convolutions 3x3 de 8 filtres, chacune suivie d'un max-pooling, puis une couche dense de 20 neurones avec dropout."""

    def __init__(self):
        super().__init__()
        def etage(c_in):
            return [nn.Conv2d(c_in, 8, 3, 1, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2)]
        # 13 x 75 -> 6 x 37 -> 3 x 18
        self.corps = nn.Sequential(*etage(1), *etage(8), nn.Flatten(), nn.Linear(8 * 3 * 18, 20), nn.ReLU(), nn.Dropout(0.5))
        self.sortie = nn.Linear(20, 1)

    def forward(self, x):
        return self.sortie(self.corps(x.unsqueeze(1))).squeeze(1)


class Noman(nn.Module):
    """CNN 2D sur la carte MFCC, dans l'esprit de la branche 2D de Noman et al. (2019) : trois blocs convolution 3x3,
    normalisation, ReLU, max-pooling (16, 32 puis 64 filtres), moyenne globale, dropout, couche finale.
    Adapté à des fenêtres de 2,5 s (l'article travaille sur un battement)."""

    def __init__(self):
        super().__init__()
        def etage(c_in, c_out):
            return [nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2)]
        self.corps = nn.Sequential(*etage(1, 16), *etage(16, 32), *etage(32, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                   nn.Dropout(0.3))
        self.sortie = nn.Linear(64, 1)

    def forward(self, x):
        return self.sortie(self.corps(x.unsqueeze(1))).squeeze(1)


class Deng(nn.Module):
    """CRNN sur les MFCC, dans l'esprit de Deng et al. (2020) : deux convolutions 2D (16 puis 32 filtres, pooling 2x2) qui
    extraient des caractéristiques locales, une LSTM bidirectionnelle (32 unités) qui suit leur évolution dans le temps,
    moyenne sur le temps puis couche finale."""

    def __init__(self):
        super().__init__()
        def etage(c_in, c_out):
            return [nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2)]
        # 13 x 75 -> 6 x 37 -> 3 x 18 : 18 instants de 32 x 3 = 96 valeurs
        self.conv = nn.Sequential(*etage(1, 16), *etage(16, 32))
        self.lstm = nn.LSTM(32 * 3, 32, batch_first=True, bidirectional=True)
        self.sortie = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv(x.unsqueeze(1))                        # (lot, 32, 3, 18)
        x = x.flatten(1, 2).transpose(1, 2)                  # (lot, 18 instants, 96)
        return self.sortie(self.lstm(x)[0].mean(dim=1)).squeeze(1)


RESEAUX = {"tcresnet8": TCResNet8, "rubin": Rubin, "noman": Noman, "deng": Deng}


def compter(modele):
    """Renvoie (nombre de paramètres, nombre de MACs) pour une fenêtre. Les MACs comptent les convolutions, la LSTM et
    les couches linéaires ; la normalisation (BatchNorm) et les activations sont ignorées."""
    macs = []

    def hook(m, entree, sortie):
        if isinstance(m, nn.Conv1d):
            # sortie : (lot, canaux, instants)
            macs.append(sortie.shape[1] * sortie.shape[2] * m.in_channels * m.kernel_size[0])
        elif isinstance(m, nn.Conv2d):
            # sortie : (lot, canaux, hauteur, largeur)
            macs.append(sortie[0].numel() * m.in_channels * m.kernel_size[0] * m.kernel_size[1])
        elif isinstance(m, nn.LSTM):
            # 4 portes, par instant et par sens : entrée -> porte et état caché -> porte
            sens = 2 if m.bidirectional else 1
            macs.append(entree[0].shape[1] * sens * 4 * m.hidden_size * (m.input_size + m.hidden_size))
        else:
            macs.append(m.in_features * m.out_features)

    hooks = [m.register_forward_hook(hook) for m in modele.modules()
             if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.LSTM, nn.Linear))]
    modele.eval()
    with torch.no_grad():
        modele(torch.zeros(1, 13, 75, device=next(modele.parameters()).device))  # une fenêtre vide suffit pour compter
    for h in hooks:
        h.remove()
    return sum(p.numel() for p in modele.parameters()), sum(macs)


if __name__ == "__main__":
    for nom, reseau in RESEAUX.items():
        p, m = compter(reseau())
        print(f"{nom:10s} {p:7d} paramètres, {m / 1e6:.2f} M MACs par fenêtre")
