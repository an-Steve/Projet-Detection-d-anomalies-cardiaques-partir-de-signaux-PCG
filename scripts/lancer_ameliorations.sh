#!/bin/bash
# Essaie deux variantes d'entraînement de TC-ResNet8 en LODO (6 bases) sur 3 NOUVELLES graines (20, 21, 22),
# avec la référence relancée sur les mêmes graines pour une comparaison appariée.
# Les graines 10, 11, 12 ont servi à confirmer le modèle : on ne les réutilise pas pour choisir.
# Usage (depuis la racine du projet) : nohup bash scripts/lancer_ameliorations.sh > ameliorations.log 2>&1 &
F=results/ameliorations.csv
for graine in 20 21 22; do
  for base in a b c d e f; do
    echo "=== base graine=$graine lodo_$base"
    python -m pcg.train --pli lodo_$base --graine $graine --fichier $F
    echo "=== aug graine=$graine lodo_$base"
    python -m pcg.train --pli lodo_$base --graine $graine --fichier $F --augmenter
    echo "=== adv0.3 graine=$graine lodo_$base"
    python -m pcg.train --pli lodo_$base --graine $graine --fichier $F --adversaire 0.3
  done
done
echo FINI
