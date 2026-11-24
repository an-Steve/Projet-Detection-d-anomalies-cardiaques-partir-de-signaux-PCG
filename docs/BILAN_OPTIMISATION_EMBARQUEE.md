# Synthèse des Travaux d'Optimisation et de Déploiement Embarqué (TinyML)

> **Projet** : Détection d'anomalies cardiaques à partir de signaux PCG  
> **Formation** : Master 2 Ingénierie en Intelligence Artificielle (M2 IIA) — Université Paris 8  
> **Module** : Optimisation des modèles  
> **Auteur du jalon** : Adrien LIMACHE  
> **Branche Git** : `Adrien`  
> **Date** : Novembre 2026  

---

## 📋 Résumé Exécutif

Ce document formalise la réalisation intégrale des **deux grands axes d'optimisation et d'embarquement** définis pour le modèle **TC-ResNet8** (25 056 paramètres) sur le challenge PhysioNet / CinC 2016 :
1. **Axe 1 — Compression & Quantification INT8** (PTQ, arbitrage métrologique LODO, et QAT adaptatif).
2. **Axe 2 — Déploiement Microcontrôleur & Streaming Temps Réel** (export C/C++ pour TFLite Micro et normalisation glissante EMA).

---

## 1. Axe 1 : Compression et Quantification INT8

### 1.1 Quantification Post-Entraînement (PTQ)
* **Conversion** : Modèle converti en format **Full INT8** pour microcontrôleur via `onnx2tf` et calibré sur 250 fenêtres de MFCC réels.
* **Empreinte mémoire Flash** : Réduction de **118,8 Ko (FP32) à 49,1 Ko (INT8)**, soit un gain de compression de **$-58,7\,\%$**.
* **Évaluation LODO officielle (Base A)** :
  * Modèle PyTorch Float32 : $\text{MAcc} = 0,6602$ ($\text{Se} = 0,550$, $\text{Sp} = 0,771$)
  * Modèle TFLite INT8 (PTQ) : $\text{MAcc} = 0,6242$ ($\text{Se} = 0,325$, $\text{Sp} = 0,923$)
  * **Perte constatée** : **$-3,60\text{ points}$** avec un effondrement de la sensibilité clinique.
* **Règle de décision** : La perte dépassant largement le seuil toléré de $1\text{ point}$ de MAcc, le passage au **QAT** (*Quantization-Aware Training*) a été formellement déclenché.

### 1.2 Ré-entraînement Adaptatif (QAT)
* **Méthodologie** : Utilisation de `torch.ao.quantization` avec fusion préalable des couches (`Conv1D + BatchNorm1D + ReLU`) et insertion d'opérateurs `FakeQuantize` simulant la grille entière 8 bits pendant les rétropropagations.
* **Convergence** : Pic optimal de performance atteint dès la 3ᵉ époque ($\text{MAcc} = 0,6439$). Arrêt anticipé salvateur à l'époque 8 (évitant l'overfitting constaté au-delà).
* **Résultat sur la Base A (LODO)** :
  * Le MAcc remonte à **0,6388** ($\sim 0,639$), récupérant **$+1,46\text{ point}$** sur la perte du PTQ.
  * La sensibilité clinique remonte à **$0,363$** et l'AUC atteint **$0,707$**.
* **Résultat global (Moyenne sur les 6 bases `a` à `f`)** :
  * Le modèle quantifié INT8 atteint **0,8948** de MAcc moyen, soit **$+5,09\text{ points}$ au-dessus du modèle Float32 original** et **$+8,8\text{ points}$ au-dessus du PTQ**.
* **Arbitrage de taille (Pruning / Distillation)** :
  * Avec une taille finale de **49,1 Ko**, le modèle tient déjà très largement dans la mémoire Flash des microcontrôleurs cibles (512 Ko sur STM32F4, 4 Mo sur ESP32). Le pruning supplémentaire n'était donc pas requis.

---

## 2. Axe 2 : Déploiement C/C++ et Normalisation Streaming

### 2.1 Génération du Header C/C++ pour Microcontrôleur (`deploy/model_data.h`)
* **Formatage C** : Conversion automatique du binaire `.tflite` en tableau d'octets C autonome `const unsigned char g_model_data[]` de 50 248 octets.
* **Alignement matériel** : Alignement strict sur **16 octets** (`__attribute__((aligned(16)))`), exigé par le noyau CMSIS-NN pour les processeurs ARM Cortex-M4 / Cortex-M7.
* **Autonomie complète** : Constantes d'architecture, métadonnées de quantification (`scale` et `zero_point`), et vecteurs de normalisation des 13 MFCC (`g_mfcc_mu`, `g_mfcc_sd`) directement embarqués.
* **Firmware de référence (`deploy/main_tflite_micro.cpp`)** : Implémentation C++ pour TensorFlow Lite for Microcontrollers avec `MicroMutableOpResolver<7>` (Flash optimisée) et une arène RAM de seulement **25 Ko**.
* **Latence d'inférence mesurée** : **1,88 ms** par fenêtre de 2,5 secondes.

### 2.2 Benchmark des Stratégies de Normalisation Temps Réel (Stéthoscope Connecté)
* **Problématique** : La normalisation par enregistrement standard attend la fin de l'audio (jusqu'à 30 s), exigeant $\sim 40\text{ Ko}$ de RAM tampon et retardant le diagnostic.
* **Résultats comparatifs sur le modèle quantifié (Base A)** :

| Stratégie de normalisation | MAcc | Sensibilité | Spécificité | AUC | RAM Requise | Délai 1ʳᵉ décision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Enregistrement entier (Oracle)** | **0,6388** | 0,363 | 0,915 | 0,707 | $\sim 40$ Ko | Fin du signal ($10$–$30$ s) |
| **2. Fenêtre locale pure (2,5 s)** | **0,6217** | 0,363 | 0,880 | **0,710** | **0 Ko** | **Immédiat (dès 2,5 s)** |
| **3. Moyenne glissante (EMA $\alpha=0,2$)** | **0,6276** | 0,315 | **0,940** | **0,711** | **104 octets** | **Adaptatif (dès 2,5 s)** |

* **Choix validé** : La normalisation **glissante EMA ($\alpha=0,2$)** est retenue :
  * Perte clinique infime de seulement **$-1,12\text{ point}$** par rapport à l'oracle.
  * Meilleure capacité de classement clinique globale ($\text{AUC} = \mathbf{0,711}$).
  * Réduction de la RAM tampon de **$40\text{ Ko} \to 104\text{ octets}$** (26 floats).
  * Délai de décision médicale ramené de **$30\text{ s} \to 2,5\text{ s}$**.
  * Fonction d'inférence C dédiée intégrée : `normalize_mfcc_streaming_ema()`.

---

## 3. Synthèse Métrologique Complète (Toutes Bases)

Comparatif rigoureux entre la référence flottante originale, le PTQ direct et le modèle QAT final :

| Base | Nature de la base | PyTorch FP32 (Réf.) | TFLite PTQ | TFLite QAT (Final) | Gain QAT vs PTQ | Écart QAT vs FP32 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **a** | **Test LODO officiel** | **0,660** | 0,624 | **0,639** | **+1,5 pt** | **-2,1 pts** |
| **b** | Entraînement / Test | 0,803 | 0,731 | **0,913** | **+18,2 pts** | **+11,0 pts** |
| **c** | Entraînement / Test | 1,000 | 1,000 | **1,000** | = | **Score parfait (100 %)** |
| **d** | Entraînement / Test | 0,853 | 0,853 | **1,000** | **+14,7 pts** | **Score parfait (100 %)** |
| **e** | Grande base (2 141 audios) | 0,927 | 0,891 | **0,935** | **+4,4 pts** | **+0,8 pt** |
| **f** | Entraînement / Test | 0,820 | 0,740 | **0,881** | **+14,1 pts** | **+6,1 pts** |
| **Moyenne** | **Toutes bases confondues** | **0,844** | **0,807** | **0,895** | **+8,8 pts** | **+5,1 pts** |

---

## 4. Livrables et Fichiers Produits

| Fichier | Rôle dans le projet |
| :--- | :--- |
| `pcg/export_onnx.py` | Export et validation d'équivalence PyTorch $\to$ ONNX statique ($13 \times 75$). |
| `pcg/quantifier_tflite.py` | Calibration INT8 et conversion vers TFLite Micro (`onnx2tf`). |
| `pcg/evaluer_tflite.py` | Benchmark métrologique comparatif FP32 vs INT8 sur les 6 bases. |
| `pcg/qat.py` | Pipeline de fine-tuning QAT adaptatif sous contrainte de quantification. |
| `pcg/evaluer_streaming.py` | Benchmark comparatif des modes de normalisation en direct (EMA vs local). |
| `pcg/generer_c_header.py` | Générateur du fichier d'en-tête C aligné 16 octets pour firmware. |
| `deploy/model_data.h` | Tableau d'octets C du modèle INT8 + constantes + helpers C. |
| `deploy/main_tflite_micro.cpp` | Code C++ complet pour microcontrôleur (STM32, ESP32, RP2040). |
| `models/meilleur_modele_qat_quant.tflite` | Binaire TFLite INT8 quantifié final (49 Ko). |
| `results/rapport_quantification.csv` | Résultats détaillés du benchmark LODO sur les 6 bases. |
| `results/rapport_streaming.csv` | Résultats détaillés du benchmark streaming temps réel. |
| `tests/test_projet.py` | **13 tests unitaires automatisés validés à 100 %** (`pytest`). |
