# Classification de sons cardiaques (PCG) avec un petit réseau

Projet de M2 IIA, Optimisation des modèles. Base : PhysioNet/CinC Challenge 2016 (3 240 enregistrements, 6 bases de a à f).
Objectif : classer un enregistrement en « normal » ou « anormal » avec un réseau assez petit pour un microcontrôleur.

## Membres du groupe

| Nom | Prénom |
| --- | ------ |
| ANTON NELCON     |    Steve    |
| MOULAI HACENE   |     RANIA   |
| LIMACHE |Adrien|

## Utilisation

```bash
pip install -r requirements.txt
python donnees.py               # vérifie la lecture de la base
python mfcc.py                  # calcule les MFCC (une seule fois)
python train.py --pli strat_0   # entraîne et teste sur un pli
python -m pytest                # tests
```

Plis possibles : `strat_0` à `strat_4` (aléatoires) ou `lodo_a` à `lodo_f` (une base entière en test).
Chaque exécution ajoute une ligne à `results.csv`.

## Les fichiers

| Fichier | Rôle |
|---|---|
| `donnees.py` | lit la base et fabrique les plis |
| `mfcc.py` | signal vers MFCC (13 × 75 par fenêtre de 2,5 s) |
| `modele.py` | TC-ResNet8 (16 169 paramètres, 0,28 M MACs) |
| `train.py` | entraînement et métrique MAcc |
| `test_projet.py` | tests |
| `DECISIONS.docx` | les choix faits et leurs raisons |

## Ce qui a été fait

1. **Prétraitement** : 1 000 Hz, filtre 20-400 Hz (bande des bruits du coeur), fenêtres de 2,5 s, 13 MFCC.
   J'ai comparé avec des fenêtres de 3 s et 5 s (`results_fenetres.csv`) : 2,5 s donne un résultat équivalent et coûte moins cher, je l'ai gardé.
2. **Modèle** : TC-ResNet8, un réseau de convolutions 1D conçu pour l'audio embarqué.
3. **Entraînement** : entropie croisée pondérée (seulement 20 % d'anormaux) et arrêt anticipé.
4. **Métrique** : MAcc = (sensibilité + spécificité) / 2, calculée par enregistrement.

## Résultats

| Découpage | MAcc test |
|---|---|
| Plis aléatoires (`strat_0`) | 0,87 |
| Une base entière en test (`lodo_b`) | 0,52 |

Le 0,87 est sans doute trop optimiste : la base ne fournit pas d'identifiant patient, donc un même patient peut être à la fois dans le train et dans le test.
Le 0,52 est proche du hasard : le réseau devine la base d'origine dans 96 % des cas, ce qui laisse penser qu'il reconnaît la source plus que la maladie.
Les organisateurs du challenge obtiennent aussi un score faible (0,59) avec ce découpage.

## Limites

- 3 240 enregistrements ici, contre 3 153 annoncés dans le sujet (écart non expliqué).
- Une seule graine par résultat. Les bases c (31 enregistrements) et d (55) donnent des scores très instables.
- Reste à faire : modèle TC-ResNet-PCG, comparaison de modèles, compression, déploiement sur la carte.
