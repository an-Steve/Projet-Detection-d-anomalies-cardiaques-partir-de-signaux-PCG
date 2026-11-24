"""Quantification Post-Entraînement (PTQ) INT8 vers TensorFlow Lite pour microcontrôleur.

Ce script :
  1. Génère un jeu de données de calibration représentatif (200 fenêtres de MFCC réels).
  2. Convertit le modèle ONNX en format TensorFlow Lite quantifié Full INT8 (TFLite Micro).
  3. Mesure le gain de compression mémoire (121 Ko FP32 -> ~25-30 Ko INT8).
  4. Permet l'inférence autonome sur un fichier audio .wav via le runtime TFLite.

Usage :
    python -m pcg.quantifier_tflite
    python -m pcg.quantifier_tflite --predire a0001
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from .donnees import RACINE
from .export_onnx import resoudre_wav
from .mfcc import decouper, mfcc, pretraiter
from .train import MODELES, SEUIL, normaliser_enregistrements

DOSSIER_TFLITE = MODELES / "tflite"
FICHIER_CALIBRATION = RACINE / "data" / "calibration_data.npy"


def preparer_calibration(nb_fenetres=250, modele_meta=None):
    """Génère un échantillon représentatif de fenêtres MFCC normalisées pour la calibration INT8."""
    fichier_features = RACINE / "data" / "mfcc_2.5" / "features.npy"
    fichier_fenetres = RACINE / "data" / "mfcc_2.5" / "fenetres.csv"
    fichier_meta = Path(modele_meta) if modele_meta else (MODELES / "meilleur_modele.meta.json")

    if not fichier_features.exists():
        raise FileNotFoundError(f"Caractéristiques introuvables : {fichier_features}. Lancez 'python -m pcg.mfcc'.")

    # Charger les caractéristiques et appliquer la normalisation par enregistrement (Niveau 2)
    features = np.load(fichier_features)
    if fichier_fenetres.exists():
        import pandas as pd
        fen = pd.read_csv(fichier_fenetres)
        features = normaliser_enregistrements(features, fen.id.to_numpy())

    indices = np.linspace(0, len(features) - 1, nb_fenetres, dtype=int)
    echantillon = np.array(features[indices], dtype=np.float32)

    # Normalisation avec mu et sd du modèle (Niveau 3)
    if fichier_meta.exists():
        with open(fichier_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        mu = np.array(meta["mu"], dtype=np.float32).reshape(1, 13, 1)
        sd = np.array(meta["sd"], dtype=np.float32).reshape(1, 13, 1)
        echantillon = (echantillon - mu) / sd

    # Forme adaptée pour TFLite : (N, 1, 13, 75)
    echantillon = echantillon[:, None, :, :].astype(np.float32)
    FICHIER_CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
    np.save(FICHIER_CALIBRATION, echantillon)
    return FICHIER_CALIBRATION


def charger_interpreteur_tflite(modele_path):
    """Charge le runtime TFLite (via ai_edge_litert ou tflite_runtime ou tensorflow)."""
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter(model_path=str(modele_path))
    except ImportError:
        pass
    try:
        import tflite_runtime.interpreter as tflite
        return tflite.Interpreter(model_path=str(modele_path))
    except ImportError:
        pass
    try:
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=str(modele_path))
    except ImportError:
        raise ImportError(
            "Aucun interpréteur TFLite disponible. Installez onnx2tf : pip install onnx2tf"
        )


def quantifier_vers_tflite(
    modele_onnx=MODELES / "meilleur_modele.onnx",
    sortie_dossier=DOSSIER_TFLITE,
    nom_cible=None,
):
    """Exécute la conversion ONNX vers TFLite Full INT8 avec onnx2tf."""
    modele_onnx = Path(modele_onnx)
    sortie_dossier = Path(sortie_dossier)
    sortie_dossier.mkdir(parents=True, exist_ok=True)

    if not modele_onnx.exists():
        from .export_onnx import exporter_onnx
        print(f"Modèle ONNX introuvable ({modele_onnx}), export en cours...")
        exporter_onnx(sortie_onnx=modele_onnx)

    # Préparation du jeu de calibration
    print(f"Préparation du jeu de données de calibration (250 fenêtres)...")
    fichier_meta = modele_onnx.with_suffix(".meta.json")
    calib_path = preparer_calibration(modele_meta=fichier_meta if fichier_meta.exists() else None)

    try:
        import onnx2tf
    except ImportError:
        raise ImportError(
            "Le convertisseur 'onnx2tf' n'est pas encore installé.\n"
            "Pour l'installer, lancez : pip install onnx2tf"
        )

    print(f"Conversion et quantification INT8 en cours via onnx2tf...")
    onnx2tf.convert(
        input_onnx_file_path=str(modele_onnx),
        output_folder_path=str(sortie_dossier),
        output_integer_quantized_tflite=True,
        keep_ncw_or_nchw_or_ncdhw_input_names=["entree_mfcc"],
        custom_input_op_name_np_data_path=[
            ["entree_mfcc", str(calib_path), [0.0], [1.0]]
        ],
        verbosity="error",
    )

    # Identifier le fichier TFLite quantifié produit correspondant au modèle ONNX (Full INT8 en priorité)
    pattern = f"{modele_onnx.stem}*full_integer_quant.tflite"
    fichiers_tflite = sorted(sortie_dossier.glob(pattern))
    if not fichiers_tflite:
        fichiers_tflite = sorted(sortie_dossier.glob(f"{modele_onnx.stem}*integer_quant.tflite"))
    if not fichiers_tflite:
        fichiers_tflite = sorted(sortie_dossier.glob(f"{modele_onnx.stem}*.tflite"))
    if not fichiers_tflite:
        fichiers_tflite = sorted(sortie_dossier.glob("*full_integer_quant.tflite"))

    if fichiers_tflite:
        tflite_quant = fichiers_tflite[0]
        nom_final = nom_cible or (f"{modele_onnx.stem}_quant.tflite" if "meilleur_modele" not in modele_onnx.stem or "qat" in modele_onnx.stem else "meilleur_modele_quant.tflite")
        cible_finale = MODELES / nom_final
        import shutil
        shutil.copy2(tflite_quant, cible_finale)
        return cible_finale

    raise FileNotFoundError(f"Aucun fichier .tflite généré dans {sortie_dossier}")


def predire_tflite(wav_path, modele_tflite=MODELES / "meilleur_modele_quant.tflite"):
    """Prédit la probabilité d'anomalie d'un enregistrement WAV en utilisant le modèle TFLite quantifié."""
    modele_tflite = Path(modele_tflite)
    wav_path = resoudre_wav(wav_path)
    fichier_meta = MODELES / "meilleur_modele.meta.json"

    with open(fichier_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    mu = np.array(meta["mu"], dtype=np.float32).reshape(1, 13, 1)
    sd = np.array(meta["sd"], dtype=np.float32).reshape(1, 13, 1)
    seuil = meta.get("seuil", SEUIL)

    interp = charger_interpreteur_tflite(modele_tflite)
    interp.allocate_tensors()

    input_details = interp.get_input_details()[0]
    output_details = interp.get_output_details()[0]

    fs, x = wavfile.read(wav_path)
    fenetres = decouper(pretraiter(x, fs))
    X = np.stack([mfcc(f) for f in fenetres]).astype(np.float32)
    X = normaliser_enregistrements(X, np.zeros(len(X)))
    X_norm = ((X - mu) / sd).astype(np.float32)

    scores = []
    scale, zero_point = input_details.get("quantization", (0.0, 0))

    for i in range(len(X_norm)):
        fenetre = X_norm[i:i + 1]  # forme (1, 13, 75)

        # Si l'entrée du modèle TFLite est en INT8
        if input_details["dtype"] == np.int8 and scale > 0:
            fenetre = np.round(fenetre / scale + zero_point).clip(-128, 127).astype(np.int8)

        interp.set_tensor(input_details["index"], fenetre)
        interp.invoke()
        sortie = interp.get_tensor(output_details["index"])

        # Si la sortie est quantifiée en INT8
        out_scale, out_zp = output_details.get("quantization", (0.0, 0))
        if output_details["dtype"] == np.int8 and out_scale > 0:
            logit = (sortie.astype(np.float32) - out_zp) * out_scale
        else:
            logit = sortie.astype(np.float32)

        proba = float(1.0 / (1.0 + np.exp(-logit.flatten()[0])))
        scores.append(proba)

    proba_moyenne = float(np.mean(scores))
    return proba_moyenne, proba_moyenne >= seuil


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantification Post-Entraînement INT8 vers TFLite")
    parser.add_argument("--predire", default=None, help="Prédit sur un fichier audio avec le modèle TFLite INT8")
    parser.add_argument("--modele_tflite", default=str(MODELES / "meilleur_modele_quant.tflite"), help="Modèle TFLite")
    args = parser.parse_args()

    if args.predire:
        chemin_tflite = Path(args.modele_tflite)
        if not chemin_tflite.exists():
            print(f"Modèle TFLite quantifié introuvable ({chemin_tflite}). Lancement automatique de la quantification...")
            chemin_tflite = quantifier_vers_tflite()
        proba, anormal = predire_tflite(args.predire, chemin_tflite)
        verdict = "anormal" if anormal else "normal"
        print(f"Fichier : {args.predire}")
        print(f"Probabilité d'anomalie (TFLite INT8) : {proba:.3f} -> {verdict}")
    else:
        print("=== Quantification INT8 vers TFLite (TFLite Micro) ===")
        modele_final = quantifier_vers_tflite()
        taille_pt = (MODELES / "meilleur_modele.pt").stat().st_size / 1024
        taille_onnx = (MODELES / "meilleur_modele.onnx").stat().st_size / 1024
        taille_tflite = Path(modele_final).stat().st_size / 1024

        print("\n=== Synthèse de Compression Mémoire ===")
        print(f"1. PyTorch FP32  : {taille_pt:6.1f} Ko (100.0 %)")
        print(f"2. ONNX FP32     : {taille_onnx:6.1f} Ko ({taille_onnx/taille_pt*100:5.1f} %)")
        print(f"3. TFLite INT8   : {taille_tflite:6.1f} Ko ({taille_tflite/taille_pt*100:5.1f} %)")
        print(f"-> Réduction totale d'empreinte mémoire : {(1 - taille_tflite/taille_pt)*100:.1f} % !")
        print(f"-> Modèle prêt pour microcontrôleur : {modele_final}")
