"""Prédit normal / anormal pour un enregistrement .wav avec le modèle sauvegardé.

Usage : python -m pcg.predire enregistrement.wav [models/meilleur_modele.pt]
"""
import sys

import numpy as np
import torch
from scipy.io import wavfile

from .mfcc import decouper, mfcc, pretraiter
from .train import DEV, MODELES, SEUIL, charger, normaliser_enregistrements, probas


def predire(wav, modele_pt=MODELES / "meilleur_modele.pt"):
    """Probabilité que l'enregistrement soit anormal (>= SEUIL, 0,5 : anormal)."""
    modele, mu, sd, _ = charger(modele_pt)
    fs, x = wavfile.read(wav)
    X = np.stack([mfcc(f) for f in decouper(pretraiter(x, fs))]).astype(np.float32)
    X = normaliser_enregistrements(X, np.zeros(len(X)))  # un seul enregistrement : toutes les fenêtres dans le même groupe
    return float(probas(modele, torch.from_numpy((X - mu) / sd).float().to(DEV)).mean())


if __name__ == "__main__":
    p = predire(*sys.argv[1:3])
    print(f"probabilité d'anomalie : {p:.2f} -> {'anormal' if p >= SEUIL else 'normal'}")
