"""Résultats du modèle retenu, protocole par protocole, base par base.

Usage : python -m pcg.rapport     (lit results/resultats.csv et results/fenetres.csv, écrit results/rapport_*.csv)

Tableaux :
  1. les deux protocoles côte à côte (plis aléatoires et une base entière en test)
  2. le LODO détaillé par base de test, avec la taille et la part d'anormaux de la base
  3. le choix de la durée de fenêtre (2,5 s, 3 s ou 5 s)
  4. peut-on deviner la base d'origine d'un enregistrement à partir de ses MFCC ? (explique l'écart entre protocoles)
  5. et 6. comparaison des réseaux (si plusieurs sont dans resultats.csv)
  7. variantes d'entraînement (si ameliorations.csv existe)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .donnees import RACINE, lire
from .mfcc import dossier

DOSSIER = RACINE / "results"
MESURES = ["macc_test", "se", "sp", "auc"]


def resume(d, par):
    """Moyenne ± écart-type de chaque mesure (et nombre d'exécutions), regroupée par `par`."""
    g = d.groupby(par)[MESURES]
    m, s = g.mean(), g.std()
    out = pd.DataFrame({c: m[c].map("{:.3f}".format) + " ± " + s[c].map("{:.3f}".format) for c in MESURES})
    out.insert(0, "n", g.size())
    return out.rename(columns={"macc_test": "MAcc", "se": "Se", "sp": "Sp", "auc": "AUC"})


def par_base(d, df):
    """LODO par base : taille, part d'anormaux et résultats."""
    t = resume(d[d.protocole == "lodo"].assign(base=lambda x: x.pli.str[-1]), "base")
    t.insert(0, "% anormaux", (100 * (df.label == 1).groupby(df.base).mean()).round(0).astype(int))
    t.insert(0, "enregistrements", df.groupby("base").size())
    return t


def base_devinee(df):
    """Précision (moyenne par base) d'un classifieur qui devine la base d'origine à partir de la moyenne et de l'écart-type
    des MFCC de l'enregistrement. Le hasard donne 1/6 = 0,167."""
    feats = np.load(dossier() / "features.npy")
    fen = pd.read_csv(dossier() / "fenetres.csv")
    stats = pd.DataFrame(np.hstack([feats.mean(axis=2), feats.std(axis=2)])).groupby(fen.id.to_numpy()).mean()
    base = fen.drop_duplicates("id").set_index("id").base.loc[stats.index]
    pred = cross_val_predict(RandomForestClassifier(200, class_weight="balanced", random_state=0, n_jobs=-1), stats, base,
                             cv=StratifiedKFold(5, shuffle=True, random_state=0))
    return balanced_accuracy_score(base, pred)


def afficher(titre, tableau, nom):
    print(f"\n=== {titre} ===")
    print(tableau.to_string())
    tableau.to_csv(DOSSIER / f"rapport_{nom}.csv")


if __name__ == "__main__":
    df, tout = lire(), pd.read_csv(DOSSIER / "resultats.csv")
    tout["modele"] = tout["modele"].fillna("tcresnet8") if "modele" in tout else "tcresnet8"  # anciennes lignes : TC-ResNet8
    d = tout[tout.modele == "tcresnet8"]
    afficher("1. Deux protocoles (graines x plis)", resume(d, "protocole"), "protocoles")
    afficher("2. LODO : base de test", par_base(d, df), "bases")
    f = pd.read_csv(DOSSIER / "fenetres.csv")
    afficher("3. Durée de fenêtre (ancien modèle de base : 2 graines, 3 plis aléatoires, bases a et b)",
             resume(f, ["fenetre_s", "protocole"]), "fenetres")
    print(f"\n=== 4. Deviner la base à partir des MFCC : exactitude équilibrée = {base_devinee(df):.3f} (hasard : 0.167) ===")
    if tout.modele.nunique() > 1:  # comparaison : mêmes plis, mêmes graines, même procédure pour tous
        t = resume(tout, ["modele", "protocole"])
        t.insert(1, "params", tout.groupby(["modele", "protocole"]).params.first())
        afficher("5. Comparaison des réseaux (même procédure)", t, "modeles")
        lodo = tout[tout.protocole == "lodo"].assign(base=lambda x: x.pli.str[-1])
        afficher("6. LODO par base et par réseau (MAcc)",
                 lodo.pivot_table(index="base", columns="modele", values="macc_test").round(3), "modeles_bases")
    amel = DOSSIER / "ameliorations.csv"
    if amel.exists():  # variantes d'entraînement : mêmes graines et mêmes bases que la référence, comparaison appariée
        v = pd.read_csv(amel)
        p = v.pivot_table(index=["pli", "graine"], columns="variante", values="macc_test")
        t = resume(v, "variante")
        for c in p.columns.drop("base", errors="ignore"):
            x = (p[c] - p["base"]).dropna()
            t.loc[c, "écart apparié"] = f"{x.mean():+.3f} ± {x.std() / len(x) ** 0.5:.3f} (n={len(x)})"
        afficher("7. Variantes d'entraînement en LODO (graines 20-22), écart avec la référence", t, "ameliorations")
