"""Prétraitement du son et calcul des MFCC.

Étapes :
  1. rééchantillonnage de 2000 à 1000 Hz (les bruits du coeur sont graves)
  2. filtre passe-bande 20-400 Hz
  3. normalisation du signal (moyenne 0, écart-type 1)
  4. découpage en fenêtres de 2,5 s qui se chevauchent à moitié
  5. MFCC de chaque fenêtre : matrice 13 coefficients x 75 instants

Tout est codé avec numpy (sans librairie audio) pour pouvoir le réécrire en C sur le microcontrôleur.

Usage : python mfcc.py  (à lancer une fois, écrit les résultats dans data/)
"""
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, resample_poly
from donnees import BASE, DOSSIER, lire

FS = 1000                  # fréquence après rééchantillonnage (Hz)
BANDE = (20, 400)          # bande de fréquences gardée (Hz)
FENETRE, PAS = 2500, 1250  # fenêtre de 2,5 s et pas de 1,25 s, en points
N_FFT, PAS_FFT = 128, 32   # tranches de 128 points décalées de 32 : 75 tranches par fenêtre
N_MEL, N_MFCC = 26, 13     # 26 filtres Mel, 13 coefficients gardés


def pretraiter(x, fs_orig=2000):
    """Signal brut -> signal à 1000 Hz, filtré et normalisé."""
    x = resample_poly(np.asarray(x, dtype=np.float64), FS, fs_orig)
    # sosfiltfilt filtre dans les deux sens, donc sans décalage dans le temps
    x = sosfiltfilt(butter(3, BANDE, btype="band", fs=FS, output="sos"), x)
    return (x - x.mean()) / (x.std() + 1e-10)  # 1e-10 évite une division par zéro


def decouper(x):
    """Découpe le signal en fenêtres de 2,5 s. La fin incomplète est ignorée."""
    n = (len(x) - FENETRE) // PAS + 1
    return np.stack([x[i * PAS: i * PAS + FENETRE] for i in range(n)])


def _mel(f):
    return 2595 * np.log10(1 + f / 700)  # Hz -> Mel


def _hz(m):
    return 700 * (10 ** (m / 2595) - 1)  # Mel -> Hz


def filtres_mel():
    """Banque de 26 filtres triangulaires espacés sur l'échelle Mel. Forme (26, 65)."""
    freqs = np.linspace(0, FS / 2, N_FFT // 2 + 1)
    bords = _hz(np.linspace(_mel(BANDE[0]), _mel(BANDE[1]), N_MEL + 2))
    f = np.zeros((N_MEL, len(freqs)))
    for i in range(N_MEL):
        g, c, d = bords[i:i + 3]  # bords gauche, centre et droit du triangle
        f[i] = np.maximum(0, np.minimum((freqs - g) / (c - g), (d - freqs) / (d - c)))
    return f


def matrice_dct():
    """Matrice de la DCT qui réduit les 26 énergies Mel à 13 coefficients."""
    k, n = np.arange(N_MFCC)[:, None], np.arange(N_MEL)[None, :]
    d = np.sqrt(2 / N_MEL) * np.cos(np.pi * k * (2 * n + 1) / (2 * N_MEL))
    d[0] /= np.sqrt(2)
    return d


# calculés une seule fois car ils ne changent pas
FILTRES, DCT, HANN = filtres_mel(), matrice_dct(), np.hanning(N_FFT + 1)[:-1]


def mfcc(fenetre):
    """Une fenêtre de 2500 points -> matrice MFCC (13 x 75)."""
    n = (len(fenetre) - N_FFT) // PAS_FFT + 1
    # tranches de 128 points, pondérées par une fenêtre de Hann
    cadres = np.stack([fenetre[i * PAS_FFT: i * PAS_FFT + N_FFT] for i in range(n)]) * HANN
    puissance = np.abs(np.fft.rfft(cadres, axis=1)) ** 2                   # spectre de puissance (FFT)
    log_mel = 10 * np.log10(np.maximum(FILTRES @ puissance.T, 1e-10))      # filtres Mel puis log
    return DCT @ log_mel                                                   # DCT


if __name__ == "__main__":
    sortie = DOSSIER / "data"
    sortie.mkdir(exist_ok=True)
    feats, index = [], []
    for r in lire().itertuples():
        fs, x = wavfile.read(BASE / f"training-{r.base}" / f"{r.id}.wav")
        for f in decouper(pretraiter(x, fs)):
            feats.append(mfcc(f).astype(np.float32))
            index.append((r.id, r.base, r.label))  # enregistrement d'origine de chaque fenêtre
    feats = np.stack(feats)
    np.save(sortie / "features.npy", feats)
    pd.DataFrame(index, columns=["id", "base", "label"]).to_csv(sortie / "fenetres.csv", index=False)
    print("features.npy :", feats.shape)  # (nombre de fenêtres, 13, 75)
