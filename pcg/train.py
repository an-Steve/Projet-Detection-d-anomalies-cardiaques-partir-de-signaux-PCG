"""Entraînement et évaluation (TC-ResNet8 par défaut).   Exemple : python -m pcg.train --pli lodo_b

Procédure (justifications dans le README) :
  - normalisation en trois niveaux : signal (pcg/mfcc.py), MFCC de chaque enregistrement (moyenne 0, écart-type 1 par
    coefficient), puis moyenne et écart-type du train seulement ;
  - poids : chaque groupe (base, classe) pèse pareil, et chaque enregistrement pèse pareil dans son groupe ;
  - validation (arrêt anticipé) : une base entière absente du train ;
  - décision : probabilité d'un enregistrement = moyenne de ses fenêtres ; anormal si elle atteint 0,5 ;
  - métrique : MAcc = (sensibilité + spécificité) / 2, par enregistrement.

Plis : lodo_a..lodo_f, strat_0..strat_4, ou tout (modèle final : train = bases b à f, validation = base a).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .donnees import RACINE, lire, plis
from .mfcc import DUREE, dossier
from .modele import RESEAUX, compter
from .variantes import Inversion, augmenter

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU s'il y en a un
EPOQUES, PATIENCE, LOT, LR, WD = 30, 5, 128, 1e-3, 1e-4  # époques max, patience de l'arrêt anticipé, taille de lot, pas d'apprentissage, weight decay
SEUIL = 0.5
RESULTATS = RACINE / "results" / "resultats.csv"
PREDICTIONS = RACINE / "results" / "predictions"  # probabilités par enregistrement, un CSV par exécution
MODELES = RACINE / "models"                       # poids sauvegardés, un fichier .pt par exécution


def macc(y, pred):
    """MAcc = (sensibilité + spécificité) / 2. Étiquettes : 1 = anormal, -1 = normal.

    Répondre "toujours normal" donne 0,5 au lieu de 80 % de précision.
    """
    se = np.mean(pred[y == 1] == 1)    # sensibilité : anormaux bien détectés
    sp = np.mean(pred[y == -1] == -1)  # spécificité : normaux bien reconnus
    return (se + sp) / 2, se, sp


def metriques(y, p):
    """{macc, se, sp, auc} à partir des étiquettes et des probabilités. L'AUC ne dépend pas du seuil."""
    m, se, sp = macc(y, np.where(p >= SEUIL, 1, -1))
    auc = roc_auc_score(y == 1, p) if len(set(y)) == 2 else float("nan")
    return dict(macc=float(m), se=float(se), sp=float(sp), auc=float(auc))


def normaliser_enregistrements(feats, ids):
    """Chaque enregistrement : MFCC recentrés (moyenne 0) puis mis à l'échelle (écart-type 1), coefficient par coefficient."""

    def par_enreg(a):  # pour chaque fenêtre : moyenne de ses coefficients sur toutes les fenêtres de son enregistrement, forme (n, 13, 1)
        return pd.DataFrame(a).groupby(ids).transform("mean").to_numpy()[:, :, None]

    feats = feats - par_enreg(feats.mean(axis=2))
    return feats / np.sqrt(par_enreg((feats ** 2).mean(axis=2)) + 1e-8)


def probas(modele, X):
    """Probabilité d'anormalité de chaque fenêtre (lots de 1024 pour limiter la mémoire)."""
    modele.eval()
    with torch.no_grad():
        return torch.cat([torch.sigmoid(modele(X[i:i + 1024])) for i in range(0, len(X), 1024)]).cpu().numpy()


def evaluer(modele, X, ids, label):
    """(métriques, probabilité de chaque enregistrement), calculées par enregistrement."""
    p = pd.Series(probas(modele, X)).groupby(ids).mean()  # moyenne des fenêtres de chaque enregistrement
    return metriques(label[p.index].to_numpy(), p.to_numpy()), p


def ajuster(Xtr, ytr, wtr, Xva, idva, label, reseau="tcresnet8", aug=False, adv=0.0, btr=None):
    """Entraîne un réseau (voir modele.RESEAUX) avec arrêt anticipé. Renvoie (modèle, meilleur MAcc validation).

    Deux options pour essayer d'améliorer le LODO (désactivées par défaut, le résultat de référence ne change pas) :
      aug : perturbe les MFCC pendant l'entraînement (voir augmenter).
      adv : entraînement adversaire. Une petite tête essaie de deviner la base d'origine à partir des caractéristiques du
            réseau, et le gradient inversé (poids adv) pousse le réseau à les rendre inutiles pour cela. btr = base de
            chaque fenêtre d'entraînement (indice entier). La tête n'existe que pendant l'entraînement : le modèle
            sauvegardé est le même TC-ResNet8.
    """
    modele = RESEAUX[reseau]().to(DEV)
    params = list(modele.parameters())
    if adv > 0:
        tete = torch.nn.Linear(modele.sortie.in_features, int(btr.max()) + 1).to(DEV)
        params += list(tete.parameters())
    opt = torch.optim.Adam(params, lr=LR, weight_decay=WD)
    meilleur, etat, attente = -1, None, 0
    for ep in range(EPOQUES):
        modele.train()
        # fenêtres du train dans le désordre, par lots
        for b in torch.randperm(len(Xtr), device=DEV).split(LOT):
            opt.zero_grad()
            x = augmenter(Xtr[b]) if aug else Xtr[b]
            if adv > 0:
                carac = modele.caracteristiques(x)
                perte = F.binary_cross_entropy_with_logits(modele.sortie(carac).squeeze(1), ytr[b], reduction="none")
                perte_base = F.cross_entropy(tete(Inversion.apply(carac, adv)), btr[b], reduction="none")
                ((perte + perte_base) * wtr[b]).sum().div(wtr[b].sum()).backward()
            else:
                perte = F.binary_cross_entropy_with_logits(modele(x), ytr[b], reduction="none")
                (perte * wtr[b]).sum().div(wtr[b].sum()).backward()
            opt.step()
        score = evaluer(modele, Xva, idva, label)[0]["macc"]
        print(f"époque {ep:2d}  MAcc validation = {score:.3f}", flush=True)
        if score > meilleur:  # amélioration : on garde cette version
            meilleur, attente = score, 0
            etat = {k: v.clone() for k, v in modele.state_dict().items()}
        else:
            attente += 1
            if attente >= PATIENCE:  # pas de progrès depuis PATIENCE époques : on arrête
                break
    modele.load_state_dict(etat)  # meilleure version de la validation, pas la dernière
    return modele, meilleur


def entrainer(pli, graine=0, duree=DUREE, sauver=True, reseau="tcresnet8", aug=False, adv=0.0):
    """Entraîne et teste sur un pli. Renvoie une ligne de résultats ; sauvegarde les poids et les prédictions.
    Tous les réseaux passent par exactement les mêmes données, normalisations, poids, optimiseur et arrêt anticipé."""
    if adv > 0 and reseau != "tcresnet8":
        raise ValueError("l'entraînement adversaire n'est prévu que pour tcresnet8")
    torch.manual_seed(graine)
    feats = np.load(dossier(duree) / "features.npy")
    fen = pd.read_csv(dossier(duree) / "fenetres.csv")
    label = fen.drop_duplicates("id").set_index("id").label  # une étiquette par enregistrement
    ids_pli = plis(lire())[pli]
    feats = normaliser_enregistrements(feats, fen.id.to_numpy())

    def partie(nom):
        """Fenêtres, identifiants, étiquettes et bases de la partie 'train', 'val' ou 'test'."""
        m = fen.id.isin(ids_pli[nom]).to_numpy()
        return feats[m], fen.id[m].to_numpy(), fen.label[m].to_numpy(), fen.base[m].to_numpy()

    Xtr, idtr, ytr, btr = partie("train")
    # normalisation par coefficient MFCC, avec la moyenne et l'écart-type du train seulement
    mu, sd = Xtr.mean(axis=(0, 2), keepdims=True), Xtr.std(axis=(0, 2), keepdims=True)
    normaliser = lambda X: torch.from_numpy((X - mu) / sd).float().to(DEV)
    Xva, idva, _, _ = partie("val")
    Xte, idte, _, _ = partie("test")
    Xtr, Xva, Xte = normaliser(Xtr), normaliser(Xva), normaliser(Xte)

    # poids de chaque fenêtre : 1 / (fenêtres de son enregistrement), puis chaque groupe (base, classe) ramené à un poids total de 1
    n_fen = pd.Series(idtr).map(pd.Series(idtr).value_counts()).to_numpy()
    w = 1.0 / n_fen
    w = w / pd.Series(w).groupby(np.char.add(btr.astype(str), ytr.astype(str))).transform("sum").to_numpy()
    wtr = torch.from_numpy(w).float().to(DEV)
    ytr = torch.from_numpy((ytr == 1).astype(np.float32)).to(DEV)  # 1 = anormal, 0 = normal

    bases_train = torch.from_numpy(np.unique(btr, return_inverse=True)[1]).long().to(DEV)  # indice de la base de chaque fenêtre
    modele, meilleur = ajuster(Xtr, ytr, wtr, Xva, idva, label, reseau, aug, adv, bases_train)
    nul = dict(macc=float("nan"), se=float("nan"), sp=float("nan"), auc=float("nan"))
    m, p = evaluer(modele, Xte, idte, label) if len(Xte) else (nul, None)  # pli 'tout' : pas de test
    params, macs = compter(modele)
    variante = "+".join(v for v, actif in [("aug", aug), (f"adv{adv:g}", adv > 0)] if actif) or "base"
    res = dict(modele=reseau, variante=variante, protocole=pli.split("_")[0], pli=pli, graine=graine, fenetre_s=float(duree),
               params=params, macs=macs, macc_val=float(meilleur), macc_test=m["macc"], se=m["se"], sp=m["sp"], auc=m["auc"])
    if sauver:
        sauvegarder(res, modele, mu, sd, p, label)
    return res


def sauvegarder(res, modele, mu, sd, p, label):
    """Écrit les poids (avec la normalisation du train et le résultat) et, s'il y a un test, les probabilités des enregistrements.
    Le modèle final (pli 'tout') s'appelle models/meilleur_modele.pt ; les autres exécutions vont dans models/comparaison/."""
    reseau = res.get("modele", "tcresnet8")
    final = res["pli"] == "tout"
    nom = "meilleur_modele" if final else f"{res['pli']}_g{res['graine']}"
    if reseau != "tcresnet8":
        nom = f"{reseau}_{nom}"
    if res.get("variante", "base") != "base":  # variantes d'entraînement : ne pas écraser les résultats de référence
        nom = f"{res['variante']}_{nom}"
    dossier_poids = MODELES if final else MODELES / "comparaison"
    dossier_poids.mkdir(parents=True, exist_ok=True)
    torch.save(dict(etat={k: v.cpu() for k, v in modele.state_dict().items()},
                    mu=torch.from_numpy(mu), sd=torch.from_numpy(sd), resultat=res), dossier_poids / f"{nom}.pt")
    if p is not None:
        PREDICTIONS.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(dict(id=p.index, label=label[p.index].to_numpy(), proba=p.to_numpy())).to_csv(
            PREDICTIONS / f"{nom}.csv", index=False)


def charger(chemin):
    """Recharge un modèle sauvegardé. Renvoie (modèle en mode évaluation, mu, sd, résultat de l'exécution).

    Pour prédire : MFCC -> normaliser_enregistrements -> (X - mu) / sd -> sigmoid(modèle) -> moyenne des fenêtres
    de l'enregistrement -> anormal si >= 0,5. pcg.predire fait tout cela à partir d'un fichier .wav.
    """
    d = torch.load(chemin, map_location=DEV, weights_only=True)
    modele = RESEAUX[d["resultat"].get("modele", "tcresnet8")]().to(DEV)  # anciens fichiers : TC-ResNet8
    modele.load_state_dict(d["etat"])
    return modele.eval(), d["mu"].numpy(), d["sd"].numpy(), d["resultat"]


def enregistrer(res, fichier=RESULTATS):
    """Ajoute une ligne de résultats au CSV (créé s'il n'existe pas)."""
    fichier.parent.mkdir(exist_ok=True)
    ancien = pd.read_csv(fichier) if fichier.exists() else pd.DataFrame()
    pd.concat([ancien, pd.DataFrame([res])]).to_csv(fichier, index=False)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pli", default="strat_0", help="lodo_a..lodo_f, strat_0..strat_4, ou tout (modèle final)")
    a.add_argument("--graine", type=int, default=0)
    a.add_argument("--modele", default="tcresnet8", choices=list(RESEAUX), help="réseau à entraîner (même procédure pour tous)")
    a.add_argument("--augmenter", action="store_true", help="perturbe les MFCC pendant l'entraînement (variante à tester)")
    a.add_argument("--adversaire", type=float, default=0.0, help="poids de l'entraînement adversaire contre la base (0 = désactivé)")
    a.add_argument("--duree", type=float, default=DUREE, help="durée des fenêtres MFCC (s), calculées avec pcg.mfcc")
    a.add_argument("--fichier", default=str(RESULTATS), help="CSV où ajouter la ligne de résultats (ameliorations.csv pour les variantes)")
    a.add_argument("--sans-sauvegarde", action="store_true", help="ne pas écrire les poids ni les prédictions")
    args = a.parse_args()
    res = entrainer(args.pli, args.graine, args.duree, sauver=not args.sans_sauvegarde, reseau=args.modele,
                    aug=args.augmenter, adv=args.adversaire)
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()})
    enregistrer(res, Path(args.fichier))
