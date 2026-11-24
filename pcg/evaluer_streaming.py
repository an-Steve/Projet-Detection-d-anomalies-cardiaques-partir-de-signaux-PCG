"""Évaluation des stratégies de normalisation temps réel pour microcontrôleur (Chantier 2).

Problématique embarquée :
    La normalisation standard par enregistrement nécessite de stocker tout l'enregistrement audio
    (10 à 30 secondes), ce qui excède la RAM disponible sur microcontrôleur et retarde la décision.

Ce script compare 3 stratégies sur le modèle TFLite INT8 QAT :
    1. Enregistrement complet (Oracle de référence - nécessite tout le signal en RAM).
    2. Fenêtre locale de 2,5 s (Zéro mémoire tampon, décision immédiate dès la première fenêtre).
    3. Moyenne glissante exponentielle (EMA, alpha=0.2, ne nécessite que 104 octets de RAM).

Usage :
    python -m pcg.evaluer_streaming [--base a] [--modele_tflite models/meilleur_modele_qat_quant.tflite]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .donnees import RACINE
from .quantifier_tflite import charger_interpreteur_tflite
from .train import MODELES, metriques, normaliser_enregistrements

FICHIER_RAPPORT_STREAMING = RACINE / "results" / "rapport_streaming.csv"


def normaliser_par_fenetre(feats):
    """Normalise chaque fenêtre de 2,5 s indépendamment (Niveau 2 local, zéro stockage)."""
    m = feats.mean(axis=2, keepdims=True)
    s = np.sqrt(((feats - m) ** 2).mean(axis=2, keepdims=True) + 1e-8)
    return (feats - m) / s


def normaliser_ema(feats, ids, alpha=0.2):
    """Normalise via moyenne mobile exponentielle (EMA), simulant un stéthoscope temps réel."""
    feats_norm = np.zeros_like(feats)
    df_ids = pd.DataFrame({"id": ids, "idx": np.arange(len(ids))})

    for _, groupe in df_ids.groupby("id", sort=False):
        indices = groupe["idx"].to_numpy()
        mu_ema = None
        var_ema = None

        for idx in indices:
            fenetre = feats[idx]  # shape (13, 75)
            m_fen = fenetre.mean(axis=1, keepdims=True)  # (13, 1)
            v_fen = ((fenetre - m_fen) ** 2).mean(axis=1, keepdims=True)  # (13, 1)

            if mu_ema is None:
                mu_ema = m_fen
                var_ema = v_fen
            else:
                mu_ema = alpha * m_fen + (1.0 - alpha) * mu_ema
                var_ema = alpha * v_fen + (1.0 - alpha) * var_ema

            s_ema = np.sqrt(var_ema + 1e-8)
            feats_norm[idx] = (fenetre - mu_ema) / s_ema

    return feats_norm


def evaluer_tflite_predictions(interp, X_norm, ids):
    """Inférence TFLite INT8 sur un lot de fenêtres et agrégation par enregistrement."""
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp = inp.get("quantization", (0.0, 0))
    out_scale, out_zp = out.get("quantization", (0.0, 0))

    scores = []
    t0 = time.perf_counter()

    for i in range(len(X_norm)):
        fen = X_norm[i:i + 1]
        if inp["dtype"] == np.int8 and in_scale > 0:
            fen = np.round(fen / in_scale + in_zp).clip(-128, 127).astype(np.int8)

        interp.set_tensor(inp["index"], fen)
        interp.invoke()
        sortie = interp.get_tensor(out["index"])

        if out["dtype"] == np.int8 and out_scale > 0:
            logit = (sortie.astype(np.float32) - out_zp) * out_scale
        else:
            logit = sortie.astype(np.float32)

        scores.append(1.0 / (1.0 + np.exp(-logit.flatten()[0])))

    latence_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(X_norm))
    p_enreg = pd.Series(scores).groupby(ids).mean()
    return p_enreg, latence_ms


def comparer_strategies(
    base_cible="a",
    modele_tflite=MODELES / "meilleur_modele_qat_quant.tflite",
    modele_meta=MODELES / "meilleur_modele_qat.meta.json",
):
    """Compare rigoureusement les 3 méthodes de normalisation sur la base spécifiée."""
    modele_tflite = Path(modele_tflite)
    if not modele_tflite.exists():
        modele_tflite = MODELES / "meilleur_modele_quant.tflite"

    modele_meta = Path(modele_meta)
    if not modele_meta.exists():
        modele_meta = MODELES / "meilleur_modele.meta.json"

    import json
    with open(modele_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    mu = np.array(meta["mu"], dtype=np.float32).reshape(1, 13, 1)
    sd = np.array(meta["sd"], dtype=np.float32).reshape(1, 13, 1)

    # Chargement des données
    fichier_features = RACINE / "data" / "mfcc_2.5" / "features.npy"
    fichier_fenetres = RACINE / "data" / "mfcc_2.5" / "fenetres.csv"

    features_toutes = np.load(fichier_features)
    fen = pd.read_csv(fichier_fenetres)

    masque = (fen.base == base_cible) if base_cible else np.ones(len(fen), dtype=bool)
    feats = features_toutes[masque]
    ids = fen.id[masque].to_numpy()
    labels = fen.drop_duplicates("id").set_index("id").label

    # Charger le runtime TFLite
    interp = charger_interpreteur_tflite(modele_tflite)
    interp.allocate_tensors()

    print(f"\n{'='*85}")
    print(f"  BENCHMARK DES STRATÉGIES DE NORMALISATION STREAMING (Modèle TFLite INT8)")
    print(f"  Base évaluée : '{base_cible}' ({len(np.unique(ids))} enregistrements, {len(feats)} fenêtres)")
    print(f"{'='*85}")

    strategies = [
        ("1. Enregistrement entier (Oracle)", lambda f, i: normaliser_enregistrements(f, i), "~40 Ko", "Fin de l'audio (~10-30 s)"),
        ("2. Fenêtre locale (2,5 s pure)", lambda f, i: normaliser_par_fenetre(f), "0 Ko", "Immédiate (dès 2,5 s)"),
        ("3. Moyenne glissante (EMA a=0.2)", lambda f, i: normaliser_ema(f, i, alpha=0.2), "104 octets", "Adaptative (dès 2,5 s)"),
    ]

    resultats = []

    for nom, fonction_norm, ram_requise, latence_decision in strategies:
        # Étape 1 : Normalisation Niveau 2
        f_norm2 = fonction_norm(feats.copy(), ids)
        # Étape 2 : Normalisation Niveau 3 (Dataset)
        f_norm3 = ((f_norm2 - mu) / sd).astype(np.float32)

        # Inférence TFLite INT8
        p_enreg, lat_ms = evaluer_tflite_predictions(interp, f_norm3, ids)
        y_vrai = labels[p_enreg.index].to_numpy()
        m = metriques(y_vrai, p_enreg.to_numpy())

        resultats.append({
            "strategie": nom,
            "base": base_cible,
            "macc": m["macc"],
            "se": m["se"],
            "sp": m["sp"],
            "auc": m["auc"],
            "ram": ram_requise,
            "latence_decision": latence_decision,
            "latence_inference_ms": lat_ms,
        })

    df = pd.DataFrame(resultats)

    # Affichage du tableau comparatif
    print("\n" + "="*95)
    print(f"{'Stratégie de normalisation':<36} | {'MAcc':^8} | {'Se':^7} | {'Sp':^7} | {'AUC':^7} | {'RAM requise':^11} | {'Délai 1re décision':^18}")
    print("-" * 95)
    for _, r in df.iterrows():
        print(f"{r['strategie']:<36} | {r['macc']:^8.4f} | {r['se']:^7.3f} | {r['sp']:^7.3f} | {r['auc']:^7.3f} | {r['ram']:^11} | {r['latence_decision']:^18}")
    print("=" * 95)

    macc_ref = df.loc[0, "macc"]
    macc_fen = df.loc[1, "macc"]
    macc_ema = df.loc[2, "macc"]

    print("\n=== Synthèse Embarquée & Recommandation ===")
    print(f"1. Perte MAcc fenêtre locale vs enregistrement entier : {(macc_fen - macc_ref)*100:+.2f} points")
    print(f"2. Perte MAcc moyenne glissante (EMA) vs enregistrement entier : {(macc_ema - macc_ref)*100:+.2f} points")

    if abs(macc_ema - macc_ref) <= 0.02:
        print("\n-> [CONCLUSION CLINIQUE & EMBARQUÉE] :")
        print("   La stratégie glissante (EMA) est VALIDÉE avec succès !")
        print("   Elle permet une détection temps réel continue (dès 2,5 s) avec seulement 104 octets de RAM,")
        print("   tout en préservant l'exactitude clinique du modèle !")

    FICHIER_RAPPORT_STREAMING.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FICHIER_RAPPORT_STREAMING, index=False)
    print(f"\n-> Rapport streaming sauvegardé dans : {FICHIER_RAPPORT_STREAMING}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation de la normalisation streaming temps réel")
    parser.add_argument("--base", default="a", help="Base à évaluer (défaut: 'a' pour le LODO officiel)")
    parser.add_argument(
        "--modele_tflite",
        default=str(MODELES / "meilleur_modele_qat_quant.tflite"),
        help="Chemin du modèle TFLite INT8",
    )
    args = parser.parse_args()

    comparer_strategies(base_cible=args.base, modele_tflite=args.modele_tflite)
