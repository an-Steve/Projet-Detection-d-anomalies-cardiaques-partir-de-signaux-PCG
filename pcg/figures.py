"""Schémas de la comparaison entre réseaux.   Usage : python -m pcg.figures   (lit results/resultats.csv, écrit results/figures/*.png)

  1. MAcc des quatre réseaux dans les deux protocoles
  2. MAcc en LODO, base par base
  3. MAcc en LODO selon le coût de calcul
  4. Profil du LODO : MAcc, sensibilité, spécificité, AUC
  5. Ablation : l'effet de chaque élément du modèle (nécessite results/archive_recherche/, sinon la figure existante est conservée)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .donnees import RACINE, lire

DOSSIER = RACINE / "results" / "figures"
NOMS = {"tcresnet8": "TC-ResNet8", "deng": "Deng (CRNN)", "noman": "Noman (CNN 2D)", "rubin": "Rubin (CNN 2D)"}
COULEURS = {"tcresnet8": "#1f5fbf", "deng": "#e08a1e", "noman": "#2a9d8f", "rubin": "#8d8d8d"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})


def hasard(ax):
    ax.axhline(0.5, color="#c0392b", ls="--", lw=1)
    ax.text(0.99, 0.5, "hasard", color="#c0392b", fontsize=8, va="bottom", ha="right", transform=ax.get_yaxis_transform())


def enregistrer(fig, nom):
    DOSSIER.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(DOSSIER / f"{nom}.png", bbox_inches="tight")
    plt.close(fig)


def protocoles(d):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, (proto, titre) in zip(axes, [("strat", "Plis aléatoires (optimiste)"), ("lodo", "Une base entière en test (LODO)")]):
        g = d[d.protocole == proto].groupby("modele").macc_test
        m, s = g.mean().reindex(NOMS), g.std().reindex(NOMS)
        ax.bar(range(4), m, yerr=s, capsize=3, color=[COULEURS[k] for k in NOMS], width=0.65, error_kw=dict(lw=1, color="#333"))
        for i, v in enumerate(m):
            ax.text(i, 0.42, f"{v:.3f}", ha="center", color="white", fontsize=9, fontweight="bold")
        ax.set_xticks(range(4), [NOMS[k].replace(" (", "\n(") for k in NOMS], fontsize=8.5)
        ax.set_title(titre, fontsize=10.5)
        hasard(ax)
    axes[0].set_ylim(0.4, 1.0)
    axes[0].set_ylabel("MAcc (moyenne ± écart-type)")
    enregistrer(fig, "1_protocoles")


def par_base(d, df):
    t = d[d.protocole == "lodo"].assign(base=lambda x: x.pli.str[-1]).pivot_table(index="base", columns="modele", values="macc_test")
    part = (100 * (df.label == 1).groupby(df.base).mean()).round().astype(int)
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    larg = 0.2
    for j, k in enumerate(NOMS):
        ax.bar(np.arange(len(t)) + (j - 1.5) * larg, t[k], larg, label=NOMS[k], color=COULEURS[k])
    ax.set_xticks(range(len(t)), [f"base {b}\n{df.base.eq(b).sum()} enreg.\n{part[b]} % anormaux" for b in t.index], fontsize=8.5)
    ax.set_ylim(0.4, 0.9)
    ax.set_ylabel("MAcc en LODO (moyenne de 3 graines)")
    hasard(ax)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15), frameon=False, fontsize=9)
    enregistrer(fig, "2_lodo_par_base")


def cout(d):
    t = d[d.protocole == "lodo"].groupby("modele").agg(macc=("macc_test", "mean"), params=("params", "first"), macs=("macs", "first"))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for k in NOMS:
        ax.scatter(t.macs[k] / 1e6, t.macc[k], s=t.params[k] / 40, color=COULEURS[k], alpha=0.85, edgecolor="white")
        ax.annotate(f"{NOMS[k]}\n{t.params[k]:,} param.".replace(",", " "), (t.macs[k] / 1e6, t.macc[k]),
                    textcoords="offset points", xytext=(0, 20 if k != "tcresnet8" else 24), ha="center", fontsize=8.5)
    ax.set_xlabel("Coût de calcul (millions de MACs par fenêtre) : moins = mieux")
    ax.set_ylabel("MAcc en LODO : plus = mieux")
    ax.set_xlim(0, 2.6)
    ax.set_ylim(0.48, 0.7)
    ax.text(0.02, 0.97, "Taille des disques = nombre de paramètres", transform=ax.transAxes, va="top", fontsize=8, color="#555")
    hasard(ax)
    enregistrer(fig, "3_precision_et_cout")


def profil(d):
    g = d[d.protocole == "lodo"].groupby("modele")[["macc_test", "se", "sp", "auc"]].mean().reindex(NOMS)
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    larg = 0.2
    for j, k in enumerate(NOMS):
        v = g.loc[k]
        ax.bar(np.arange(4) + (j - 1.5) * larg, v, larg, label=NOMS[k], color=COULEURS[k])
        for i, x in enumerate(v):
            ax.text(i + (j - 1.5) * larg, x + 0.01, f"{x:.2f}", ha="center", fontsize=7)
    ax.set_xticks(range(4), ["MAcc", "Sensibilité", "Spécificité", "AUC"])
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Valeur en LODO (moyenne)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15), frameon=False, fontsize=9)
    enregistrer(fig, "4_profil_lodo")


def ablation():
    """Retire un élément à la fois de la configuration complète (LODO, 6 bases x 3 graines, largeur x1,0), d'après la recherche
    archivée. Le départ est mesuré sur d'autres graines (10, 11, 12)."""
    j = pd.read_csv(RACINE / "results" / "archive_recherche" / "recherche_journal.csv")
    c = pd.read_csv(RACINE / "results" / "archive_recherche" / "recherche_confirmation.csv")
    complet = j[(j.facteur == "norm") & (j.valeur == "cmvn")].macc.iloc[0]
    ligne = lambda f, v: j[(j.facteur == f) & (j.valeur.astype(str) == v)].macc.iloc[0]
    depart = c[(c.protocole == "lodo") & (c.norm == "global") & ~c.poids_base].macc_test.mean()
    barres = [("Modèle de départ\n(aucun des trois)", depart, "#8d8d8d"),
              ("Sans normalisation\npar enreg.", ligne("norm", "global"), "#c0392b"),
              ("Sans poids par base", ligne("poids_base", "False"), "#9db8e0"),
              ("Sans validation\npar base entière", ligne("val_base", "False"), "#9db8e0"),
              ("Sans poids par\nenregistrement", ligne("ponderer", "False"), "#9db8e0"),
              ("Configuration complète", complet, "#1f5fbf")]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for i, (nom, v, col) in enumerate(barres):
        ax.bar(i, v, color=col, width=0.65)
        ax.text(i, 0.405, f"{v:.3f}", ha="center", color="white" if col != "#9db8e0" else "#222", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(barres)), [b[0] for b in barres], fontsize=8)
    ax.set_ylim(0.4, 0.7)
    ax.set_ylabel("MAcc en LODO")
    hasard(ax)
    enregistrer(fig, "5_ablation")


if __name__ == "__main__":
    d = pd.read_csv(RACINE / "results" / "resultats.csv")
    d["modele"] = d["modele"].fillna("tcresnet8") if "modele" in d else "tcresnet8"
    protocoles(d)
    par_base(d, lire())
    cout(d)
    profil(d)
    if (RACINE / "results" / "archive_recherche").exists():
        ablation()
    else:
        print("ablation : données de la recherche absentes, results/figures/5_ablation.png inchangée")
    print(f"figures écrites dans {DOSSIER}")
