"""Entraînement et évaluation du réseau.   Exemple : python train.py --pli strat_0

Déroulement :
  1. entraînement sur les fenêtres du train
  2. après chaque époque, calcul du MAcc sur la validation
  3. arrêt quand la validation ne progresse plus (early stopping), on garde la meilleure version
  4. évaluation finale sur le test, ajoutée à results.csv

Deux choix importants :
  - Les anormaux sont rares (20 %), donc leurs erreurs sont plus pénalisées (pos_weight).
  - Le réseau classe des fenêtres, mais on veut un résultat par enregistrement :
    on moyenne les probabilités de ses fenêtres et on décide "anormal" si la moyenne dépasse 0,5.
"""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from donnees import DOSSIER, lire, plis
from modele import TCResNet8


def macc(y, pred):
    """MAcc = (sensibilité + spécificité) / 2. Étiquettes : 1 = anormal, -1 = normal.

    Répondre "toujours normal" donne 0,5 au lieu de 80 % de précision.
    """
    se = np.mean(pred[y == 1] == 1)    # sensibilité : anormaux bien détectés
    sp = np.mean(pred[y == -1] == -1)  # spécificité : normaux bien reconnus
    return (se + sp) / 2, se, sp


def evaluer(modele, X, ids, label):
    """MAcc calculé par enregistrement à partir des fenêtres X (ids = enregistrement de chaque fenêtre)."""
    modele.eval()
    with torch.no_grad():
        # sigmoid : score -> probabilité ; on traite par lots de 1024 pour limiter la mémoire
        p = torch.cat([torch.sigmoid(modele(X[i:i + 1024])) for i in range(0, len(X), 1024)]).numpy()
    p = pd.Series(p).groupby(ids).mean()  # moyenne des fenêtres de chaque enregistrement
    return macc(label[p.index].to_numpy(), np.where(p.to_numpy() >= 0.5, 1, -1))


def entrainer(pli, graine=0, epoques=30, patience=5):
    torch.manual_seed(graine)
    feats = np.load(DOSSIER / "data" / "features.npy")
    fen = pd.read_csv(DOSSIER / "data" / "fenetres.csv")
    label = fen.drop_duplicates("id").set_index("id").label  # une étiquette par enregistrement
    ids_pli = plis(lire())[pli]

    def partie(nom):
        """Fenêtres, identifiants et étiquettes de la partie 'train', 'val' ou 'test'."""
        m = fen.id.isin(ids_pli[nom]).to_numpy()
        return feats[m], fen.id[m].to_numpy(), fen.label[m].to_numpy()

    Xtr, _, ytr = partie("train")
    # normalisation par coefficient MFCC, avec la moyenne et l'écart-type du train seulement
    # (pour ne pas utiliser d'information du test)
    mu, sd = Xtr.mean(axis=(0, 2), keepdims=True), Xtr.std(axis=(0, 2), keepdims=True)
    normaliser = lambda X: torch.from_numpy((X - mu) / sd).float()
    Xva, idva, _ = partie("val")
    Xte, idte, _ = partie("test")
    Xtr, Xva, Xte = normaliser(Xtr), normaliser(Xva), normaliser(Xte)
    ytr = torch.from_numpy((ytr == 1).astype(np.float32))  # 1 = anormal, 0 = normal
    # poids de la classe anormale = rapport normaux / anormaux
    poids = (ytr == 0).sum() / (ytr == 1).sum()

    modele = TCResNet8()
    opt = torch.optim.Adam(modele.parameters(), lr=1e-3, weight_decay=1e-4)
    meilleur, etat, attente = -1, None, 0
    for ep in range(epoques):
        modele.train()
        # fenêtres du train dans le désordre, par lots de 128
        for b in torch.randperm(len(Xtr)).split(128):
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(modele(Xtr[b]), ytr[b], pos_weight=poids).backward()
            opt.step()
        score = evaluer(modele, Xva, idva, label)[0]
        print(f"époque {ep:2d}  MAcc validation = {score:.3f}", flush=True)
        if score > meilleur:  # amélioration : on garde cette version
            meilleur, attente = score, 0
            etat = {k: v.clone() for k, v in modele.state_dict().items()}
        else:
            attente += 1
            if attente >= patience:  # pas de progrès depuis `patience` époques : on arrête
                break
    modele.load_state_dict(etat)  # meilleure version de la validation, pas la dernière
    m, se, sp = evaluer(modele, Xte, idte, label)
    return dict(pli=pli, graine=graine, macc_val=float(meilleur), macc_test=float(m), se=float(se), sp=float(sp))


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pli", default="strat_0", help="lodo_a..lodo_f ou strat_0..strat_4")
    a.add_argument("--graine", type=int, default=0)
    args = a.parse_args()
    res = entrainer(args.pli, args.graine)
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()})
    # on ajoute le résultat à results.csv (créé s'il n'existe pas)
    fichier = DOSSIER / "results.csv"
    ancien = pd.read_csv(fichier) if fichier.exists() else pd.DataFrame()
    pd.concat([ancien, pd.DataFrame([res])]).to_csv(fichier, index=False)
