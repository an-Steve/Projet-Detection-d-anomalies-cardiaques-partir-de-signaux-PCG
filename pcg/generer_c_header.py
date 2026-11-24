"""Générateur de code C/C++ pour le déploiement sur microcontrôleur (TFLite Micro).

Ce script convertit le modèle TFLite quantifié (INT8) en un fichier d'en-tête C
(`deploy/model_data.h`) autonome, aligné sur 16 octets, contenant :
  1. Le tableau d'octets du modèle (équivalent portable de xxd -i).
  2. Les paramètres de quantification (échelles et points zéro entrée/sortie).
  3. Les constantes de normalisation MFCC (mu et sd par canal).
  4. Les fonctions utilitaires C pour la quantification et la prédiction.

Usage :
    python -m pcg.generer_c_header [--modele models/meilleur_modele_qat_quant.tflite]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .donnees import RACINE
from .quantifier_tflite import charger_interpreteur_tflite
from .train import MODELES

DOSSIER_DEPLOY = RACINE / "deploy"


def formater_tableau_c(octets, elements_par_ligne=12):
    """Formate un flux binaire en tableau d'octets hexadécimaux C."""
    lignes = []
    for i in range(0, len(octets), elements_par_ligne):
        lot = octets[i:i + elements_par_ligne]
        lignes.append("    " + ", ".join(f"0x{b:02x}" for b in lot) + ",")
    return "\n".join(lignes)


def generer_c_header(
    modele_tflite=MODELES / "meilleur_modele_qat_quant.tflite",
    fichier_sortie=DOSSIER_DEPLOY / "model_data.h",
    arena_size_ko=25,
):
    """Génère le fichier deploy/model_data.h complet pour microcontrôleur."""
    modele_tflite = Path(modele_tflite)
    if not modele_tflite.exists():
        modele_tflite = MODELES / "meilleur_modele_quant.tflite"
    if not modele_tflite.exists():
        raise FileNotFoundError(f"Modèle TFLite introuvable : {modele_tflite}")

    fichier_sortie = Path(fichier_sortie)
    fichier_sortie.parent.mkdir(parents=True, exist_ok=True)

    print(f"Chargement du modèle TFLite : {modele_tflite}...")
    with open(modele_tflite, "rb") as f:
        octets = f.read()

    # Extraction des paramètres de quantification depuis le runtime TFLite
    interp = charger_interpreteur_tflite(modele_tflite)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    in_scale, in_zp = inp.get("quantization", (1.0, 0))
    out_scale, out_zp = out.get("quantization", (1.0, 0))

    # Chargement des paramètres de normalisation MFCC
    fichier_meta = modele_tflite.parent / f"{modele_tflite.stem.replace('_quant', '')}.meta.json"
    if not fichier_meta.exists():
        fichier_meta = MODELES / "meilleur_modele.meta.json"

    mu = [0.0] * 13
    sd = [1.0] * 13
    seuil = 0.5
    if fichier_meta.exists():
        with open(fichier_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
            mu = meta.get("mu", mu)[:13]
            sd = meta.get("sd", sd)[:13]
            seuil = meta.get("seuil", 0.5)

    mu_c_str = ", ".join(f"{float(x):.6f}f" for x in mu)
    sd_c_str = ", ".join(f"{float(x):.6f}f" for x in sd)
    octets_c_str = formater_tableau_c(octets)

    contenu = f"""/*
 * AUTOGÉNÉRÉ PAR pcg.generer_c_header - PROJET PCG (M2 IIA)
 * Modèle : TC-ResNet8 quantifié Full INT8 pour TensorFlow Lite Micro
 * Taille du modèle Flash : {len(octets)} octets ({len(octets) / 1024:.2f} Ko)
 */

#ifndef MODEL_DATA_H_
#define MODEL_DATA_H_

#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {{
#endif

/* ==================================================================== */
/* Constantes d'architecture et de quantification                       */
/* ==================================================================== */
#define MODEL_INPUT_CHANNELS        13
#define MODEL_INPUT_FRAMES          75
#define MODEL_TOTAL_INPUT_ELEMENTS  (MODEL_INPUT_CHANNELS * MODEL_INPUT_FRAMES)

#define MODEL_INPUT_SCALE           {float(in_scale):.8f}f
#define MODEL_INPUT_ZERO_POINT      {int(in_zp)}

#define MODEL_OUTPUT_SCALE          {float(out_scale):.8f}f
#define MODEL_OUTPUT_ZERO_POINT     {int(out_zp)}

#define MODEL_DECISION_THRESHOLD    {float(seuil):.2f}f
#define MODEL_TENSOR_ARENA_SIZE     ({int(arena_size_ko)} * 1024)

/* ==================================================================== */
/* Paramètres de normalisation MFCC (Niveau 3 - Dataset)                */
/* ==================================================================== */
static const float g_mfcc_mu[MODEL_INPUT_CHANNELS] = {{ {mu_c_str} }};
static const float g_mfcc_sd[MODEL_INPUT_CHANNELS] = {{ {sd_c_str} }};

/* ==================================================================== */
/* Fonctions utilitaires d'inférence en C                               */
/* ==================================================================== */

/**
 * Quantifie une valeur flottante MFCC normalisée vers l'entier signé INT8.
 */
static inline int8_t quantize_input(float real_val) {{
    int32_t q = (int32_t)roundf(real_val / MODEL_INPUT_SCALE) + MODEL_INPUT_ZERO_POINT;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    return (int8_t)q;
}}

/**
 * Dé-quantifie la sortie INT8 du réseau vers le logit continu en virgule flottante.
 */
static inline float dequantize_output(int8_t quant_val) {{
    return ((float)quant_val - (float)MODEL_OUTPUT_ZERO_POINT) * MODEL_OUTPUT_SCALE;
}}

/**
 * Calcule la sigmoïde d'un logit pour obtenir la probabilité clinique d'anomalie.
 */
static inline float compute_probability(float logit) {{
    return 1.0f / (1.0f + expf(-logit));
}}

/**
 * Indique si la probabilité dépasse le seuil clinique d'alerte.
 */
static inline bool is_abnormal(float probability) {{
    return probability >= MODEL_DECISION_THRESHOLD;
}}

/**
 * Normalise en direct un vecteur MFCC (13 coefficients) via une moyenne glissante exponentielle (EMA).
 * Ne consomme que 26 floats de RAM (104 octets) pour mu_ema et var_ema.
 */
static inline void normalize_mfcc_streaming_ema(const float* raw_mfcc, float* out_norm, float* mu_ema, float* var_ema, float alpha) {{
    for (int c = 0; c < MODEL_INPUT_CHANNELS; ++c) {{
        float val = raw_mfcc[c];
        mu_ema[c] = alpha * val + (1.0f - alpha) * mu_ema[c];
        float diff = val - mu_ema[c];
        var_ema[c] = alpha * (diff * diff) + (1.0f - alpha) * var_ema[c];
        float std = sqrtf(var_ema[c] + 1e-8f);
        float norm_l2 = diff / std;
        out_norm[c] = (norm_l2 - g_mfcc_mu[c]) / g_mfcc_sd[c];
    }}
}}

/* ==================================================================== */
/* Tableau binaire du modèle TFLite (aligné sur 16 octets pour Cortex-M) */
/* ==================================================================== */
#if defined(__GNUC__) || defined(__clang__)
#define ALIGNAS16 __attribute__((aligned(16)))
#elif defined(_MSC_VER)
#define ALIGNAS16 __declspec(align(16))
#else
#define ALIGNAS16
#endif

ALIGNAS16 static const unsigned char g_model_data[] = {{
{octets_c_str}
}};

static const unsigned int g_model_data_len = {len(octets)};

#ifdef __cplusplus
}}
#endif

#endif /* MODEL_DATA_H_ */
"""
    with open(fichier_sortie, "w", encoding="utf-8") as f:
        f.write(contenu)

    print(f"\n{'='*70}")
    print(f"  GÉNÉRATION DU C HEADER RÉUSSIE !")
    print(f"  Fichier C généré      : {fichier_sortie}")
    print(f"  Taille du modèle      : {len(octets):,d} octets ({len(octets) / 1024:.2f} Ko)")
    print(f"  Alignement mémoire    : 16 octets (requis pour CMSIS-NN / TFLite Micro)")
    print(f"  Échelle entrée INT8   : {in_scale:.6f} (zero-point: {in_zp})")
    print(f"  Échelle sortie INT8   : {out_scale:.6f} (zero-point: {out_zp})")
    print(f"{'='*70}")
    return fichier_sortie


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère deploy/model_data.h pour microcontrôleur")
    parser.add_argument(
        "--modele",
        default=str(MODELES / "meilleur_modele_qat_quant.tflite"),
        help="Chemin du modèle TFLite source",
    )
    parser.add_argument(
        "--sortie",
        default=str(DOSSIER_DEPLOY / "model_data.h"),
        help="Chemin du fichier header C de destination",
    )
    args = parser.parse_args()

    generer_c_header(modele_tflite=args.modele, fichier_sortie=args.sortie)
