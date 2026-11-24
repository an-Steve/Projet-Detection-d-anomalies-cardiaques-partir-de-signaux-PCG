"""Quantization-Aware Training (QAT) officiel avec PyTorch (torch.ao.quantization).

Procédure :
  1. Charge les poids déjà entraînés de 'models/meilleur_modele.pt'.
  2. Fusionne les couches Conv1D + BatchNorm1D + ReLU pour l'embarqué.
  3. Insère les couches FakeQuantize simulant la grille INT8 (backend qnnpack/ARM).
  4. Ré-entraîne pendant quelques époques (fine-tuning, lr=1e-4) sur les bases b-f avec validation sur la base a.
  5. Exporte les poids QAT et génère le modèle final TFLite INT8 recalibré.

Usage :
    python -m pcg.qat [--epoques 8] [--lr 1e-4]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.ao.quantization as tq
import torch.nn.functional as F

from .donnees import RACINE, lire, plis
from .export_onnx import exporter_onnx
from .mfcc import DUREE, dossier
from .modele import TCResNet8
from .quantifier_tflite import DOSSIER_TFLITE, quantifier_vers_tflite
from .train import (
    DEV,
    LOT,
    MODELES,
    PATIENCE,
    charger,
    evaluer,
    metriques,
    normaliser_enregistrements,
    probas,
)


def preparer_modele_qat(modele):
    """Prépare un modèle TC-ResNet8 pour le QAT en fusionnant les couches et insérant les observateurs INT8."""
    modele.train()
    modele.qconfig = tq.get_default_qat_qconfig("qnnpack")

    # Fusion des couches : Conv1d + BatchNorm1d + ReLU
    modele = tq.fuse_modules_qat(modele, [["entree.0", "entree.1", "entree.2"]])
    for i in range(3):
        modele = tq.fuse_modules_qat(
            modele,
            [
                [f"blocs.{i}.conv.0", f"blocs.{i}.conv.1", f"blocs.{i}.conv.2"],
                [f"blocs.{i}.conv.3", f"blocs.{i}.conv.4"],
                [f"blocs.{i}.raccourci.0", f"blocs.{i}.raccourci.1"],
            ],
        )

    modele_prepare = tq.prepare_qat(modele)
    return modele_prepare


def extraire_state_dict_propre(etat_qat):
    """Mappe un state_dict QAT (couches fusionnées ConvBnReLU) vers un state_dict standard TCResNet8."""
    # Si le modèle est déjà dans le format standard, rien à faire
    if "entree.1.weight" in etat_qat and "entree.0.bn.weight" not in etat_qat:
        return etat_qat

    clean = TCResNet8()
    sd = clean.state_dict()

    # Couche entrée
    sd["entree.0.weight"] = etat_qat["entree.0.weight"].clone()
    sd["entree.1.weight"] = etat_qat["entree.0.bn.weight"].clone()
    sd["entree.1.bias"] = etat_qat["entree.0.bn.bias"].clone()
    sd["entree.1.running_mean"] = etat_qat["entree.0.bn.running_mean"].clone()
    sd["entree.1.running_var"] = etat_qat["entree.0.bn.running_var"].clone()
    if "entree.0.bn.num_batches_tracked" in etat_qat:
        sd["entree.1.num_batches_tracked"] = etat_qat["entree.0.bn.num_batches_tracked"].clone()

    for i in range(3):
        # conv 0 (fused conv + bn + relu)
        sd[f"blocs.{i}.conv.0.weight"] = etat_qat[f"blocs.{i}.conv.0.weight"].clone()
        sd[f"blocs.{i}.conv.1.weight"] = etat_qat[f"blocs.{i}.conv.0.bn.weight"].clone()
        sd[f"blocs.{i}.conv.1.bias"] = etat_qat[f"blocs.{i}.conv.0.bn.bias"].clone()
        sd[f"blocs.{i}.conv.1.running_mean"] = etat_qat[f"blocs.{i}.conv.0.bn.running_mean"].clone()
        sd[f"blocs.{i}.conv.1.running_var"] = etat_qat[f"blocs.{i}.conv.0.bn.running_var"].clone()
        if f"blocs.{i}.conv.0.bn.num_batches_tracked" in etat_qat:
            sd[f"blocs.{i}.conv.1.num_batches_tracked"] = etat_qat[f"blocs.{i}.conv.0.bn.num_batches_tracked"].clone()

        # conv 3 (fused conv + bn)
        sd[f"blocs.{i}.conv.3.weight"] = etat_qat[f"blocs.{i}.conv.3.weight"].clone()
        sd[f"blocs.{i}.conv.4.weight"] = etat_qat[f"blocs.{i}.conv.3.bn.weight"].clone()
        sd[f"blocs.{i}.conv.4.bias"] = etat_qat[f"blocs.{i}.conv.3.bn.bias"].clone()
        sd[f"blocs.{i}.conv.4.running_mean"] = etat_qat[f"blocs.{i}.conv.3.bn.running_mean"].clone()
        sd[f"blocs.{i}.conv.4.running_var"] = etat_qat[f"blocs.{i}.conv.3.bn.running_var"].clone()
        if f"blocs.{i}.conv.3.bn.num_batches_tracked" in etat_qat:
            sd[f"blocs.{i}.conv.4.num_batches_tracked"] = etat_qat[f"blocs.{i}.conv.3.bn.num_batches_tracked"].clone()

        # raccourci (fused conv + bn)
        sd[f"blocs.{i}.raccourci.0.weight"] = etat_qat[f"blocs.{i}.raccourci.0.weight"].clone()
        sd[f"blocs.{i}.raccourci.1.weight"] = etat_qat[f"blocs.{i}.raccourci.0.bn.weight"].clone()
        sd[f"blocs.{i}.raccourci.1.bias"] = etat_qat[f"blocs.{i}.raccourci.0.bn.bias"].clone()
        sd[f"blocs.{i}.raccourci.1.running_mean"] = etat_qat[f"blocs.{i}.raccourci.0.bn.running_mean"].clone()
        sd[f"blocs.{i}.raccourci.1.running_var"] = etat_qat[f"blocs.{i}.raccourci.0.bn.running_var"].clone()
        if f"blocs.{i}.raccourci.0.bn.num_batches_tracked" in etat_qat:
            sd[f"blocs.{i}.raccourci.1.num_batches_tracked"] = etat_qat[f"blocs.{i}.raccourci.0.bn.num_batches_tracked"].clone()

    # Sortie
    sd["sortie.weight"] = etat_qat["sortie.weight"].clone()
    sd["sortie.bias"] = etat_qat["sortie.bias"].clone()

    return sd


def finaliser_export_qat(modele_pt=MODELES / "meilleur_modele_qat.pt"):
    """Convertit un checkpoint QAT au format TCResNet8 standard et génère ONNX + TFLite INT8."""
    modele_pt = Path(modele_pt)
    if not modele_pt.exists():
        raise FileNotFoundError(f"Checkpoint QAT introuvable : {modele_pt}")

    d = torch.load(modele_pt, map_location="cpu", weights_only=False)
    etat_propre = extraire_state_dict_propre(d["etat"])
    d["etat"] = etat_propre
    torch.save(d, modele_pt)
    print(f"-> Checkpoint converti au format TCResNet8 standard : {modele_pt}")

    sortie_onnx = MODELES / "meilleur_modele_qat.onnx"
    print(f"Export vers ONNX : {sortie_onnx}...")
    exporter_onnx(modele_pt=modele_pt, sortie_onnx=sortie_onnx)

    sortie_tflite = MODELES / "meilleur_modele_qat_quant.tflite"
    print(f"Génération du modèle TFLite INT8 recalibré : {sortie_tflite}...")
    quantifier_vers_tflite(
        modele_onnx=sortie_onnx,
        sortie_dossier=DOSSIER_TFLITE,
        nom_cible="meilleur_modele_qat_quant.tflite",
    )

    print(f"\n{'='*70}")
    print(f"  QAT ET CONVERSION TFLITE INT8 TERMINÉES AVEC SUCCÈS !")
    print(f"  Modèle final TFLite INT8 prêt : {sortie_tflite}")
    print(f"{'='*70}")
    return sortie_tflite


def entrainer_qat(
    pli="tout",
    graine=0,
    epoques=8,
    lr=1e-4,
    modele_source=MODELES / "meilleur_modele.pt",
    duree=DUREE,
):
    """Exécute le fine-tuning QAT sur les données d'entraînement avec arrêt anticipé sur la validation."""
    torch.manual_seed(graine)
    np.random.seed(graine)

    modele_source = Path(modele_source)
    print(f"\n{'='*70}")
    print(f"  QUANTIZATION-AWARE TRAINING (QAT) - PyTorch torch.ao.quantization")
    print(f"{'='*70}")
    print(f"Chargement du modèle source : {modele_source}")
    modele_base, mu, sd, _ = charger(modele_source)

    # Préparation QAT
    modele_qat = preparer_modele_qat(modele_base).to(DEV)
    print("-> Couches fusionnées (Conv1D + BN + ReLU) et FakeQuantize INT8 insérées.")

    # Chargement des caractéristiques MFCC
    print("Chargement des données MFCC...")
    feats = np.load(dossier(duree) / "features.npy")
    fen = pd.read_csv(dossier(duree) / "fenetres.csv")
    label = fen.drop_duplicates("id").set_index("id").label
    ids_pli = plis(lire())[pli]
    feats = normaliser_enregistrements(feats, fen.id.to_numpy())

    def partie(nom):
        m = fen.id.isin(ids_pli[nom]).to_numpy()
        return feats[m], fen.id[m].to_numpy(), fen.label[m].to_numpy(), fen.base[m].to_numpy()

    Xtr, idtr, ytr, btr = partie("train")
    normaliser = lambda X: torch.from_numpy(((X - mu) / sd).astype(np.float32)).to(DEV)
    Xva, idva, _, _ = partie("val")
    Xtr, Xva = normaliser(Xtr), normaliser(Xva)

    # Pondération équilibrée des fenêtres
    n_fen = pd.Series(idtr).map(pd.Series(idtr).value_counts()).to_numpy()
    w = 1.0 / n_fen
    w = w / pd.Series(w).groupby(np.char.add(btr.astype(str), ytr.astype(str))).transform("sum").to_numpy()
    wtr = torch.from_numpy(w).float().to(DEV)
    ytr = torch.from_numpy((ytr == 1).astype(np.float32)).to(DEV)

    # Optimiseur avec pas d'apprentissage réduit (fine-tuning)
    opt = torch.optim.Adam(modele_qat.parameters(), lr=lr, weight_decay=1e-5)

    print(f"\nDébut du ré-entraînement QAT ({epoques} époques max, lot={LOT}, lr={lr})...")
    meilleur, etat_opt, attente = -1.0, None, 0

    for ep in range(1, epoques + 1):
        modele_qat.train()
        for b in torch.randperm(len(Xtr), device=DEV).split(LOT):
            opt.zero_grad()
            out = modele_qat(Xtr[b])
            perte = F.binary_cross_entropy_with_logits(out, ytr[b], reduction="none")
            (perte * wtr[b]).sum().div(wtr[b].sum()).backward()
            opt.step()

        # Évaluation sur la base de validation (base a)
        modele_qat.eval()
        with torch.no_grad():
            score_val = evaluer(modele_qat, Xva, idva, label)[0]["macc"]

        print(f"  Époque {ep:2d}/{epoques:2d} | MAcc Validation (Base a) = {score_val:.4f}", flush=True)

        if score_val > meilleur:
            meilleur = score_val
            attente = 0
            etat_opt = {k: v.clone() for k, v in modele_qat.state_dict().items()}
        else:
            attente += 1
            if attente >= PATIENCE:
                print(f"  -> Arrêt anticipé : aucun progrès depuis {PATIENCE} époques.")
                break

    modele_qat.load_state_dict(etat_opt)
    modele_qat.eval().cpu()

    # Conversion des couches fusionnées vers le format standard TCResNet8
    etat_propre = extraire_state_dict_propre(etat_opt)

    # Sauvegarde des poids QAT recalibrés (format TCResNet8 standard)
    sortie_pt = MODELES / "meilleur_modele_qat.pt"
    torch.save(
        {
            "modele": "tcresnet8",
            "etat": etat_propre,
            "mu": torch.from_numpy(mu),
            "sd": torch.from_numpy(sd),
            "resultat": {"modele": "tcresnet8", "qat": True, "macc_val": meilleur},
        },
        sortie_pt,
    )
    print(f"\n-> Poids QAT PyTorch sauvegardés dans : {sortie_pt}")

    # Export ONNX et génération TFLite INT8
    return finaliser_export_qat(sortie_pt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantization-Aware Training (QAT) de TC-ResNet8")
    parser.add_argument("--epoques", type=int, default=8, help="Nombre maximal d'époques de fine-tuning")
    parser.add_argument("--lr", type=float, default=1e-4, help="Pas d'apprentissage pour le fine-tuning")
    parser.add_argument("--graine", type=int, default=0, help="Graine aléatoire")
    parser.add_argument(
        "--recup",
        action="store_true",
        help="Finalise l'export ONNX et TFLite depuis le checkpoint QAT déjà calculé (sans ré-entraîner)",
    )
    args = parser.parse_args()

    if args.recup:
        finaliser_export_qat()
    else:
        entrainer_qat(epoques=args.epoques, lr=args.lr, graine=args.graine)
