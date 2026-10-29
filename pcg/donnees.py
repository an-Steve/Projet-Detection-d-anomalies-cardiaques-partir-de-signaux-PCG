"""Lecture de la base et découpage en train / validation / test.

Trois types de découpage (chacun est un dictionnaire d'ensembles d'identifiants d'enregistrements) :
  - lodo_x  : une base entière (a à f) sert de test. La validation est une autre base entière (a, ou b si le test est a)
              et le train garde le reste. Mesure la généralisation à une source jamais vue.
  - strat_i : 5 plis aléatoires qui gardent la même proportion de normaux et d'anormaux dans chaque base ;
              15 % du train sert de validation. Optimiste : un même patient peut être dans le train et dans le test.
  - tout    : modèle final. Train = bases b à f, validation = base a (jamais dans le train), pas de test.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

RACINE = Path(__file__).resolve().parent.parent
BASE = next(RACINE.glob("data/classification-of-heart-sound*"), None)  # base PhysioNet décompressée
if BASE is None:
    raise FileNotFoundError(f"dossier PhysioNet 'classification-of-heart-sound...' introuvable dans {RACINE / 'data'}")


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
    """Renvoie {nom du pli: {"train", "val", "test"}}."""

    def par_base(test, val):  # test et validation = bases entières, train = le reste
        return {"train": set(df[~df.base.isin([test, val])].id), "val": set(df[df.base == val].id),
                "test": set(df[df.base == test].id)}

    def aleatoire(train, test):  # 15 % du train sert de validation
        tr, va = train_test_split(train.id, test_size=0.15, stratify=train.strate, random_state=42)
        return {"train": set(tr), "val": set(va), "test": set(test.id)}

    res = {f"lodo_{b}": par_base(b, "b" if b == "a" else "a") for b in sorted(df.base.unique())}
    for i, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(df, df.strate)):
        res[f"strat_{i}"] = aleatoire(df.iloc[tr], df.iloc[te])
    res["tout"] = par_base("", "a")
    return res


if __name__ == "__main__":
    # contrôle rapide : nombre d'enregistrements et d'anormaux par base
    df = lire()
    print(len(df), "enregistrements,", f"{100 * (df.label == 1).mean():.1f} % d'anormaux")
    print(df.groupby("base").label.agg(n="size", anormaux=lambda s: (s == 1).sum()))
