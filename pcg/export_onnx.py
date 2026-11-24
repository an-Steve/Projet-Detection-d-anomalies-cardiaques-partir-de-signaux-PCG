"""Export et validation du modèle TC-ResNet8 au format ONNX standard.

Usage :
    python -m pcg.export_onnx [--modele models/meilleur_modele.pt] [--sortie models/meilleur_modele.onnx]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from scipy.io import wavfile

from .donnees import RACINE
from .mfcc import decouper, mfcc, pretraiter
from .train import MODELES, SEUIL, charger, normaliser_enregistrements


def exporter_onnx(modele_pt=MODELES / "meilleur_modele.pt", sortie_onnx=None, batch_size=1, opset=13):
    """Exporte un modèle PyTorch TC-ResNet8 en format ONNX statique pour l'embarqué.

    Sauvegarde également les paramètres de normalisation (mu, sd) dans un fichier .json associé.
    """
    modele_pt = Path(modele_pt)
    if sortie_onnx is None:
        sortie_onnx = modele_pt.with_suffix(".onnx")
    else:
        sortie_onnx = Path(sortie_onnx)

    sortie_onnx.parent.mkdir(parents=True, exist_ok=True)
    modele, mu, sd, res = charger(modele_pt)
    modele.eval().cpu()

    # Forme d'entrée statique pour microcontrôleur : (batch_size, 13 coefficients MFCC, 75 trames temporelles)
    x_factice = torch.zeros(batch_size, 13, 75, dtype=torch.float32)

    torch.onnx.export(
        modele,
        x_factice,
        str(sortie_onnx),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["entree_mfcc"],
        output_names=["score_anomalie"],
        dynamo=False,
    )

    # Vérification de la conformité du modèle ONNX
    modele_charge = onnx.load(str(sortie_onnx))
    onnx.checker.check_model(modele_charge)

    # Sauvegarde des métadonnées (mu, sd et résultats d'entraînement) pour inférence autonome
    fichier_meta = sortie_onnx.with_suffix(".meta.json")
    meta = {
        "mu": mu.flatten().tolist(),
        "sd": sd.flatten().tolist(),
        "seuil": SEUIL,
        "input_shape": [batch_size, 13, 75],
        "resultats": res or {},
    }
    with open(fichier_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return sortie_onnx, fichier_meta


def valider_onnx(modele_pt=MODELES / "meilleur_modele.pt", modele_onnx=None, nb_echantillons=100, atol=1e-5):
    """Vérifie l'équivalence numérique stricte entre PyTorch et ONNX Runtime sur des entrées aléatoires."""
    modele_pt = Path(modele_pt)
    if modele_onnx is None:
        modele_onnx = modele_pt.with_suffix(".onnx")
    else:
        modele_onnx = Path(modele_onnx)

    modele, _, _, _ = charger(modele_pt)
    modele.eval().cpu()

    session = ort.InferenceSession(str(modele_onnx), providers=["CPUExecutionProvider"])
    diffs = []

    for _ in range(nb_echantillons):
        x = np.random.randn(1, 13, 75).astype(np.float32)
        with torch.no_grad():
            py_score = modele(torch.from_numpy(x)).numpy()
        ort_score = session.run(["score_anomalie"], {"entree_mfcc": x})[0]
        diffs.append(np.abs(py_score - ort_score).max())

    max_diff = float(max(diffs))
    conforme = max_diff < atol
    return conforme, max_diff


def resoudre_wav(chemin):
    """Résout le chemin d'un fichier audio (chemin complet, nom de fichier, ou identifiant comme 'a0001')."""
    p = Path(chemin)
    if p.exists() and p.is_file():
        return p
    nom = p.name
    if not nom.endswith(".wav"):
        nom = f"{nom}.wav"
    trouve = next(RACINE.glob(f"data/**/{nom}"), None)
    if trouve and trouve.is_file():
        return trouve
    raise FileNotFoundError(f"Fichier audio introuvable : '{chemin}'")


def predire_onnx(wav_path, modele_onnx=None):
    """Prédit la probabilité d'anomalie d'un enregistrement WAV en utilisant uniquement ONNX Runtime."""
    wav_path = resoudre_wav(wav_path)
    if modele_onnx is None:
        modele_onnx = MODELES / "meilleur_modele.onnx"
    else:
        modele_onnx = Path(modele_onnx)

    fichier_meta = modele_onnx.with_suffix(".meta.json")
    with open(fichier_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    mu = np.array(meta["mu"], dtype=np.float32).reshape(1, 13, 1)
    sd = np.array(meta["sd"], dtype=np.float32).reshape(1, 13, 1)
    seuil = meta.get("seuil", SEUIL)

    session = ort.InferenceSession(str(modele_onnx), providers=["CPUExecutionProvider"])
    fs, x = wavfile.read(wav_path)

    # Prétraitement et MFCC
    fenetres = decouper(pretraiter(x, fs))
    X = np.stack([mfcc(f) for f in fenetres]).astype(np.float32)
    X = normaliser_enregistrements(X, np.zeros(len(X)))
    X_norm = ((X - mu) / sd).astype(np.float32)

    # Inférence fenêtre par fenêtre (batch_size = 1 comme sur microcontrôleur)
    scores = []
    for i in range(len(X_norm)):
        fenetre = X_norm[i:i + 1]  # forme (1, 13, 75)
        logit = session.run(["score_anomalie"], {"entree_mfcc": fenetre})[0]
        proba = 1.0 / (1.0 + np.exp(-logit.item()))  # fonction sigmoïde
        scores.append(proba)

    proba_moyenne = float(np.mean(scores))
    return proba_moyenne, proba_moyenne >= seuil


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export et validation ONNX de TC-ResNet8")
    parser.add_argument("--modele", default=str(MODELES / "meilleur_modele.pt"), help="Chemin du modèle .pt source")
    parser.add_argument("--sortie", default=None, help="Chemin du fichier .onnx destination")
    parser.add_argument("--predire", default=None, help="Chemin d'un fichier .wav à classifier avec ONNX Runtime")
    args = parser.parse_args()

    if args.predire:
        modele_onnx = Path(args.sortie) if args.sortie else Path(args.modele).with_suffix(".onnx")
        if not modele_onnx.exists():
            print(f"Modèle ONNX introuvable ({modele_onnx}), export initial en cours...")
            exporter_onnx(args.modele, modele_onnx)
        proba, anormal = predire_onnx(args.predire, modele_onnx)
        verdict = "anormal" if anormal else "normal"
        print(f"Fichier : {args.predire}")
        print(f"Probabilité d'anomalie (ONNX Runtime) : {proba:.3f} -> {verdict}")
    else:
        print(f"Export de {args.modele} vers ONNX...")
        fichier_onnx, fichier_meta = exporter_onnx(args.modele, args.sortie)
        taille_pt_ko = Path(args.modele).stat().st_size / 1024
        taille_onnx_ko = fichier_onnx.stat().st_size / 1024
        print(f"-> Modèle PyTorch source : {taille_pt_ko:.1f} Ko")
        print(f"-> Modèle ONNX exporté   : {taille_onnx_ko:.1f} Ko ({fichier_onnx})")
        print(f"-> Métadonnées sauvées   : {fichier_meta}")

        print("Validation numérique (100 échantillons aléatoires)...")
        valide, max_diff = valider_onnx(args.modele, fichier_onnx)
        print(f"-> Écart maximal |PyTorch - ONNX| : {max_diff:.2e}")
        if valide:
            print("-> [SUCCÈS] Le modèle ONNX est strictement équivalent au modèle PyTorch !")
        else:
            print("-> [ATTENTION] Écart numérique supérieur au seuil toléré (1e-5).")
