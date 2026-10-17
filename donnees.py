"""Lecture de la base et découpage en train / validation / test.

Deux types de découpage :
  - lodo_x : une base entière (a à f) sert de test, les autres servent à l'entraînement.
             Permet de voir si le modèle marche sur une source jamais vue.
  - strat_i : 5 plis aléatoires qui gardent la même proportion de normaux et d'anormaux.
              Sert à comparer les modèles entre eux.
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

DOSSIER = Path(__file__).resolve().parent
# le dossier PhysioNet est dans le projet ou juste à côté
BASE = next(p for d in (DOSSIER, DOSSIER.parent) for p in d.glob("classification-of-heart-sound*"))


def lire():
    """Lit la liste des enregistrements avec leur étiquette (-1 = normal, 1 = anormal)."""
    tables = []
    for dossier in sorted(BASE.glob("training-?")):
        t = pd.read_csv(dossier / "REFERENCE.csv", header=None, names=["id", "label"])
        t["base"] = dossier.name[-1]  # lettre de la base : a, b, c, d, e ou f
        tables.append(t)
    df = pd.concat(tables, ignore_index=True)
    # strate = base + classe (ex. "a1"), pour garder les mêmes proportions dans chaque morceau
    df["strate"] = df.base + df.label.astype(str)
    return df


def plis(df):
    """Renvoie un dictionnaire {nom du pli: {"train", "val", "test"}} avec des ensembles d'identifiants."""

    def pli(train, test):
        # 15 % du train sert de validation
        tr, va = train_test_split(train.id, test_size=0.15, stratify=train.strate, random_state=42)
        return {"train": set(tr), "val": set(va), "test": set(test.id)}

    res = {}
    # lodo : une base en test, toutes les autres en train
    for b in sorted(df.base.unique()):
        res[f"lodo_{b}"] = pli(df[df.base != b], df[df.base == b])
    # plis aléatoires : chaque part sert une fois de test
    for i, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(df, df.strate)):
        res[f"strat_{i}"] = pli(df.iloc[tr], df.iloc[te])
    return res


if __name__ == "__main__":
    # contrôle rapide : nombre d'enregistrements et d'anormaux par base
    df = lire()
    print(len(df), "enregistrements,", f"{100 * (df.label == 1).mean():.1f} % d'anormaux")
    print(df.groupby("base").label.agg(n="size", anormaux=lambda s: (s == 1).sum()))
