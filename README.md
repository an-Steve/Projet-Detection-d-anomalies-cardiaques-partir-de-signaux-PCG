# Classification de sons cardiaques (PCG) avec un petit réseau
> Formation : Master 2 Ingénierie en Intelligence Artificielle - Université Paris 8  | Module : Optimisation des modèles
>

## Membres du groupe

<table>
  <tr>
    <th colspan="3" align="center">Membres du groupe</th>
  </tr>
  <tr>
    <td align="center">ANTON NELCON Steve</td>
    <td align="center">MOULAI HACENE Rania</td>
    <td align="center">LIMACHE Adrien</td>
  </tr>
</table>

## Présentation du projet

Les maladies cardiovasculaires sont la première cause de mortalité dans le monde.
Le phonocardiogramme (PCG) enregistre les vibrations sonores produites par l'activité
mécanique du cœur : c'est un moyen de dépistage non invasif, très peu coûteux, qui
nécessite seulement un capteur simple (microphone ou piézoélectrique). Il est donc bien
adapté à un dispositif portable et autonome pouvant servir de premier filtre avant
orientation vers un cardiologue, notamment dans les zones à ressources médicales limitées.

Ce projet vise à classer des enregistrements PCG en **normal / anormal** avec un modèle
suffisamment léger pour tourner sur une cible embarquée à faibles ressources
(microcontrôleur de type ARM Cortex-M, ESP32, Raspberry Pi Pico ou Raspberry Pi).

## Données

Base **PhysioNet/CinC Challenge 2016** (Clifford et al., *Classification of normal/abnormal
heart sound recordings*, Computing in Cardiology 2016) : 3 240 enregistrements répartis en 6 bases (a à f).

**Lien du téléchargement** : https://physionet.org/content/challenge-2016/1.0.0/
(à décompresser dans `data/`)

## Objectifs
1. **Modèle léger dès la conception** : choisir une architecture économe en paramètres
   et en calcul, et optimiser le prétraitement (filtrage, fenêtrage, MFCC) pour limiter
   le coût en amont.
2. **Compression et déploiement embarqué** : appliquer des techniques de compression
   (quantification, pruning, distillation, factorisation à faible rang), mesurer leur impact
   sur la précision, la mémoire et la latence, puis comparer aux travaux de l'état de l'art.

## Pipeline
Données PhysioNet → Prétraitement et segmentation → Modèle léger → Comparaison avec d'autres modèles
→ Compression → Déploiement embarqué

## Métrique
Modified Accuracy : **MAcc = (sensibilité + spécificité) / 2**, métrique officielle du
challenge. Le meilleur score du challenge avoisine 0,86. On rapporte aussi l'AUC.

## Utilisation

```bash
pip install -r requirements.txt
python -m pcg.mfcc                              # calcule les MFCC (une seule fois)
python -m pcg.train --pli lodo_b                # entraîne et teste sur un pli
python -m pcg.train --modele deng --pli lodo_b  # autre réseau (rubin, noman, deng)
python -m pcg.train --pli tout                  # modèle final -> models/meilleur_modele.pt
python -m pcg.rapport                           # tableaux de résultats -> results/rapport_*.csv
python -m pcg.figures                           # schémas -> results/figures/
python -m pcg.predire fichier.wav               # normal ou anormal pour un enregistrement
python -m pytest                                # tests
```

Plis possibles : `strat_0` à `strat_4` (aléatoires) ou `lodo_a` à `lodo_f` (une base entière en test).
Chaque exécution ajoute une ligne à `results/resultats.csv`.

## Les fichiers

| Élément | Rôle |
|---|---|
| `pcg/donnees.py` | lit la base et fabrique les plis |
| `pcg/mfcc.py` | signal vers MFCC (13 × 75 par fenêtre de 2,5 s) |
| `pcg/modele.py` | TC-ResNet8 (25 056 paramètres, 0,43 M MACs) et les trois réseaux de comparaison |
| `pcg/train.py` | entraînement et métriques (MAcc, sensibilité, spécificité, AUC) |
| `pcg/predire.py` | prédiction à partir d'un fichier .wav |
| `pcg/rapport.py`, `pcg/figures.py` | tableaux et schémas des résultats |
| `pcg/variantes.py` | deux variantes d'entraînement testées (augmentation, adversaire) |
| `tests/` | tests |
| `scripts/` | séries d'entraînements (comparaison, variantes) |
| `results/` | résultats (`resultats.csv`, `rapport_*.csv`), figures et prédictions |
| `models/` | modèle livré et poids des entraînements de comparaison |
| `docs/DECISIONS.docx` | les choix faits, leurs raisons et les essais |

## Ce qui a été fait

1. **Prétraitement** : 1 000 Hz, filtre 20-400 Hz (bande des bruits du cœur), fenêtres de 2,5 s, 13 MFCC.
   Les fenêtres de 3 s et 5 s donnent un résultat équivalent pour un coût plus élevé : 2,5 s est gardé.
2. **Normalisation par enregistrement** des MFCC : c'est l'élément qui améliore le plus le résultat entre bases (0,45 sans, 0,63 avec).
3. **Modèle** : TC-ResNet8, un réseau de convolutions 1D conçu pour l'audio embarqué (25 056 paramètres, 0,43 M MACs).
4. **Entraînement** : entropie croisée pondérée par base, classe et enregistrement, arrêt anticipé sur une base entière de validation.
5. **Comparaison avec trois autres réseaux** (Rubin, Noman, Deng) entraînés exactement de la même façon : mêmes données, plis, graines et optimiseur.
6. **Variantes testées** pour améliorer le LODO (augmentation, entraînement adversaire) : aucune ne l'améliore, la référence est conservée.

## Résultats

Moyenne sur 3 graines, calcul par enregistrement.

| Découpage | MAcc | Sensibilité | Spécificité | AUC |
|---|---|---|---|---|
| Plis aléatoires (5 plis) | 0,818 | 0,779 | 0,856 | 0,913 |
| Une base entière en test (LODO) | 0,642 | 0,687 | 0,597 | 0,709 |

Le score en plis aléatoires est optimiste : la base ne fournit pas d'identifiant patient, donc un même patient peut être à la fois dans le train et dans le test.
Le LODO est la mesure honnête : le modèle de départ y était au hasard (0,506), la normalisation par enregistrement le fait passer à 0,642.
Les organisateurs du challenge obtiennent 0,59 avec ce découpage.

**Comparaison des réseaux** (même procédure pour tous) :

| Réseau | Paramètres | MAcc plis aléatoires | MAcc LODO | AUC LODO |
|---|---|---|---|---|
| **TC-ResNet8** | 25 056 | 0,818 | **0,642** | 0,709 |
| Deng (CRNN) | 38 193 | 0,808 | 0,585 | 0,644 |
| Noman (CNN 2D) | 23 473 | 0,791 | 0,589 | 0,655 |
| Rubin (CNN 2D) | 9 361 | 0,581 | 0,543 | 0,561 |

En LODO, TC-ResNet8 gagne 17 fois sur 18 contre Deng et Noman, avec 4 à 5 fois moins de calcul. Rubin n'a pas convergé dans plusieurs entraînements : son score reflète l'entraînement, pas l'architecture. Détail dans `docs/DECISIONS.docx` et `results/figures/`.

## Limites

- 3 240 enregistrements ici, contre 3 153 annoncés dans le sujet (écart de 87 dans la base e, non expliqué). Pas d'identifiant patient.
- Le résultat reste modeste : la base f est classée presque entièrement « anormale », a et b restent autour de 0,6, et les bases c (31 enregistrements) et d (55) sont très instables.
- Le modèle livré n'a pas de vrai test et sa sensibilité est de 0,41 au seuil 0,5.
- Les réglages ont été faits en LODO sur ces mêmes bases : le résultat est un peu optimiste.
- Le prétraitement (filtre sur le signal entier, normalisation par enregistrement) devra être adapté pour tourner au fil de l'eau sur la carte.
- Reste à faire : compression, déploiement sur la carte.

## Livrables
Papier scientifique (format IMRaD), code source documenté et reproductible,
présentation orale avec démonstration.
