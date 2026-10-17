"""Le réseau TC-ResNet8 (largeur x0,5).

C'est un réseau de convolutions 1D pensé pour l'audio embarqué. Les 13 MFCC sont les canaux
d'entrée et les filtres ne glissent que dans le temps, ce qui garde le réseau petit.

Entrée : (lot, 13, 75). Sortie : un score par fenêtre (plus il est haut, plus c'est anormal).
Les canaux passent de 8 à 12, 16 puis 24.
"""
import torch
import torch.nn as nn


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
    def __init__(self):
        super().__init__()
        # première convolution : 13 MFCC -> 8 canaux
        self.entree = nn.Sequential(nn.Conv1d(13, 8, 3, 1, 1, bias=False), nn.BatchNorm1d(8), nn.ReLU())
        # trois blocs : 8 -> 12 -> 16 -> 24 canaux
        self.blocs = nn.Sequential(Bloc(8, 12), Bloc(12, 16), Bloc(16, 24))
        # couche finale : 24 valeurs -> 1 score
        self.sortie = nn.Linear(24, 1)

    def forward(self, x):
        x = self.blocs(self.entree(x))
        # moyenne sur le temps (une valeur par canal), puis score
        return self.sortie(x.mean(dim=2)).squeeze(1)


def compter(modele):
    """Renvoie (nombre de paramètres, nombre de MACs) pour une fenêtre."""
    macs = []

    def hook(m, entree, sortie):
        if isinstance(m, nn.Conv1d):
            # sortie : (lot, canaux, instants)
            macs.append(sortie.shape[1] * sortie.shape[2] * m.in_channels * m.kernel_size[0])
        else:
            macs.append(m.in_features * m.out_features)

    hooks = [m.register_forward_hook(hook) for m in modele.modules() if isinstance(m, (nn.Conv1d, nn.Linear))]
    modele.eval()
    with torch.no_grad():
        modele(torch.zeros(1, 13, 75))  # une fenêtre vide suffit pour compter
    for h in hooks:
        h.remove()
    return sum(p.numel() for p in modele.parameters()), sum(macs)


if __name__ == "__main__":
    p, m = compter(TCResNet8())
    print(f"{p} paramètres, {m / 1e6:.2f} M MACs par fenêtre")
