#!/bin/bash
# Entraîne les 3 réseaux de comparaison avec la même procédure que TC-ResNet8 :
# mêmes plis (6 LODO + 5 aléatoires), mêmes graines (10, 11, 12), mêmes hyperparamètres.
# Usage (depuis la racine du projet) : nohup bash scripts/lancer_comparaison.sh > comparaison.log 2>&1 &
for modele in rubin noman deng; do
  for graine in 10 11 12; do
    for pli in lodo_a lodo_b lodo_c lodo_d lodo_e lodo_f strat_0 strat_1 strat_2 strat_3 strat_4; do
      echo "=== $modele $pli graine $graine"
      python -m pcg.train --modele $modele --pli $pli --graine $graine
    done
  done
done
echo FINI
