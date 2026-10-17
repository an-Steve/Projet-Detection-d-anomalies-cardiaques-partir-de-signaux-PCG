# Plan d'exécution : PCG TinyML (TC-ResNet-PCG)

On avance une étape à la fois. Pour chaque étape : justifier le choix, le réaliser, le vérifier avec un test ou un chiffre, puis passer à la suivante.
Les décisions sont notées dans `DECISIONS.docx`.

## Point de départ (vérifié le 2026-10-14)
- La base PhysioNet 2016 est dans le dossier du projet : `training-a` à `training-f` (409 + 490 + 31 + 55 + 2141 + 114 = 3 240 fichiers .wav), plus `validation/`, `papers/` et `sources/`.
- Python 3.12 avec numpy, scipy, pandas, scikit-learn, torch et timm. Il manque librosa, soundfile, torchinfo, thop, statsmodels, onnx, onnxruntime, matplotlib (à vérifier), pytest et PyYAML.
- Pas de GPU en local : Colab pour Rubin et MobileViT.
- Écart à expliquer : les documents parlent de 3 153 enregistrements, le dossier en contient 3 240.
- Risque principal : la base publique n'a en général pas d'identifiant patient. Si c'est le cas, l'évaluation principale sera le leave-one-database-out.

## Étape 0 : mise en place (~1 jour)
- Créer l'arborescence `pcg-tinyml/`, faire `git init`, une branche par tâche.
- `requirements.txt` avec les versions fixées, environnement virtuel.
- Dossier `results/` pour les CSV (avec graine et pli).
- Test d'export avec un petit CNN 1D : PyTorch vers ONNX vers onnxruntime, pour repérer tôt les problèmes de conversion.
- Validation : `pytest` tourne et l'export donne les mêmes sorties que PyTorch (écart < 1e-5).

## Étape 1 : données et découpage
- À décider : identifiants patient disponibles ou non ; 5 plis `StratifiedGroupKFold` ou LODO ; taille de la validation ; que faire des enregistrements de mauvaise qualité (`REFERENCE-SQI.csv`) ; 3 153 contre 3 240.
- À faire : lire les .wav et `REFERENCE.csv` dans un tableau (id, base, label, durée, fs), écrire `splits.json` et `metrics.py` (Se, Sp, MAcc par enregistrement).
- À vérifier : statistiques de base (environ 18 % d'anormaux, durées de 5 à 120 s, fs = 2 000 Hz), aucun patient dans deux parties d'un même pli, `metrics.py` testé à la main (tout « normal » donne un MAcc de 0,5).

## Étape 2 : prétraitement MFCC
- À décider, avec des chiffres : 1 000 Hz, bande 20-400 Hz, fenêtre de 2,5 s avec 50 % de recouvrement, 13 MFCC, FFT de 128 avec un pas de 32, 26 filtres Mel. Vérifier sur 2 ou 3 enregistrements que S1/S2 et les souffles sont conservés. Les comparaisons 13 contre 26 MFCC et 2,5 s contre 5 s viendront plus tard.
- À faire : `mfcc.py` en numpy, normalisation avec les statistiques du train seulement (par pli), sortie 13 × 75.
- À vérifier : résultat proche de librosa (tolérance fixée à l'avance, par exemple 1e-4 en relatif), forme 13 × 75, pas de fuite de données.

## Étape 3 : modèles et entraînement
- À décider : TC-ResNet8 ×0,5 comme référence puis TC-ResNet-PCG ; Focal Loss ou entropie croisée pondérée ; agrégation des fenêtres (moyenne ou vote) ; 5 graines.
- À faire : `tc_resnet.py`, `tc_resnet_pcg.py` (échelles ~0,1 / 0,3 / 0,8 s avec attention sur les MFCC), `rubin.py`, et un `train.py` commun piloté par YAML avec arrêt anticipé sur la validation.
- À vérifier : paramètres et MACs mesurés avec `torchinfo` ou `thop` (estimations : ≈ 16 100 paramètres, ≈ 0,27 M MACs) ; le modèle doit pouvoir apprendre un mini-lot par coeur (perte proche de 0) ; résultats écrits en CSV.
- Seuils de décision sur le MAcc moyen des 5 plis : au moins 0,80, on continue ; entre 0,78 et 0,80, on ajuste (perte, fenêtres, largeur) ; en dessous de 0,78, plan B avec un DS-CNN 2D sur les mêmes MFCC.

## Étape 4 : comparaison et ablations
- À décider avant de voir les résultats : la règle de sélection (par exemple le plus petit modèle à moins d'un point de MAcc du meilleur, ≤ 32 Ko en INT8).
- À faire : 9 modèles avec les mêmes MFCC, plis et graines ; une ablation = une seule modification (multi-échelles, attention, 1D contre 2D, 13 contre 26 MFCC, MFCC contre log-Mel, 2,5 s contre 5 s) ; graphique MAcc en fonction des MACs ; test de McNemar.
- À vérifier : moyenne ± écart-type sur 5 graines, budgets d'entraînement équivalents.

## Étape 5 : compression
- À décider : réglages de distillation (T = 2,5 et α = 0,07, contre T = 7 et α = 0,1) comparés sur le même pli ; pruning à 20, 40 ou 60 % ; quantification post-entraînement d'abord, QAT seulement si la perte dépasse 1 point.
- À vérifier après chaque opération : MAcc, paramètres, taille, latence sur PC (CSV), et sorties ONNX INT8 comparées à PyTorch sur le jeu de test.
- Objectif : modèle INT8 sous 32 Ko avec une perte de MAcc mesurée et acceptable.

## Étape 6 : déploiement sur la carte
- À décider : quelle carte (STM32 M7/M4 ou ESP32 avec TFLite Micro) et comment mesurer l'énergie (mesure ou estimation).
- À faire : X-CUBE-AI, MFCC en C (CMSIS-DSP), mesures de latence, RAM, Flash et énergie.
- À vérifier : MFCC en C identiques à ceux de Python (avec tolérance), mêmes prédictions sur la carte et sur PC pour un petit jeu de test, latence incluant le calcul des MFCC.

## Étape 7 : rédaction et soutenance
Article IMRaD, README, figures générées depuis les CSV, diapositives et démo.

## Points à garder en tête
1. L'enseignant demande une contribution non publiée : faire une recherche bibliographique rapide (TC-ResNet et PCG, multi-échelles et attention sur les MFCC) avant l'étape 3.
2. Les chiffres des documents sont des estimations : ne les citer dans l'article qu'après les avoir mesurés.
3. Pour pouvoir refaire les expériences, mettre la graine et le pli dans chaque ligne de CSV et une config YAML par expérience.
4. Les documents répartissent le travail entre M1, M2 et M3. Ce plan est séquentiel et marche aussi si on travaille seule.
