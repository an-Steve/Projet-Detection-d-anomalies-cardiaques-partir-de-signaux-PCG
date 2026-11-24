/**
 * @file main_tflite_micro.cpp
 * @brief Exemple complet de firmware C++ pour l'inférence PCG sur microcontrôleur.
 *
 * Compatible avec :
 *   - ARM Cortex-M4 / Cortex-M7 (STM32F4, STM32H7 avec CMSIS-NN)
 *   - ESP32 / ESP32-S3 (ESP-IDF / Arduino-ESP32)
 *   - Raspberry Pi Pico (RP2040)
 *
 * Utilise la bibliothèque officielle TensorFlow Lite for Microcontrollers (TFLite Micro).
 */

#include "model_data.h"

// Inclusions officielles TensorFlow Lite Micro
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <cstdio>
#include <cmath>

namespace {
    // 1. Zone mémoire vive (Tensor Arena) réservée pour les activations intermédiaires
    // 25 Ko suffisent largement pour le réseau TC-ResNet8
    uint8_t tensor_arena[MODEL_TENSOR_ARENA_SIZE];

    // 2. Pointeurs globaux TFLite Micro
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* model_input = nullptr;
    TfLiteTensor* model_output = nullptr;
}

/**
 * @brief Initialise le modèle, l'arène mémoire et l'interpréteur TFLite Micro.
 * @return true si l'initialisation a réussi, false sinon.
 */
bool setup_tflite_micro() {
    // Initialisation matérielle spécifique à la plateforme (horloge, debug UART)
    tflite::InitializeTarget();

    // Chargement du modèle depuis la mémoire Flash (tableau d'octets g_model_data)
    model = tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("[ERREUR] Schéma TFLite incompatible ! Attendu : %d, Reçu : %ld\n",
               TFLITE_SCHEMA_VERSION, model->version());
        return false;
    }

    // Enregistrement des opérateurs nécessaires pour TC-ResNet8
    // L'utilisation de MicroMutableOpResolver permet de ne compiler QUE les opérateurs requis,
    // économisant ainsi 50 à 100 Ko de mémoire Flash supplémentaire.
    static tflite::MicroMutableOpResolver<7> micro_op_resolver;
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddAveragePool2D();
    micro_op_resolver.AddAdd();
    micro_op_resolver.AddRelu();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddLogistic(); // Pour sigmoïde si présente dans le graphe

    // Instanciation de l'interpréteur
    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, MODEL_TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    // Allocation des tenseurs d'activation dans l'arène
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        printf("[ERREUR] Échec de l'allocation des tenseurs dans l'arène de %d octets !\n",
               MODEL_TENSOR_ARENA_SIZE);
        return false;
    }

    model_input = interpreter->input(0);
    model_output = interpreter->output(0);

    printf("=== TFLite Micro initialisé avec succès ===\n");
    printf("Taille Flash modèle  : %u octets\n", g_model_data_len);
    printf("Taille Arène RAM     : %d octets\n", MODEL_TENSOR_ARENA_SIZE);
    printf("Forme entrée         : [%ld, %ld, %ld]\n",
           model_input->dims->data[0], model_input->dims->data[1], model_input->dims->data[2]);
    return true;
}

/**
 * @brief Exécute la classification sur une fenêtre de 2,5 s (13 canaux x 75 trames temporelles).
 *
 * @param mfcc_window_norm Tableau 2D aplati de taille 13x75 contenant les MFCC pré-normalisés.
 * @param out_proba Pointeur pour recevoir la probabilité calculée [0.0, 1.0].
 * @return true si le cœur est classé Anormal, false si Normal.
 */
bool classifier_fenetre_pcg(const float* mfcc_window_norm, float* out_proba) {
    if (!interpreter || !model_input || !model_output) {
        return false;
    }

    // 1. Quantification des MFCC continus vers l'entrée INT8 du modèle
    int8_t* input_buffer = model_input->data.int8;
    for (int i = 0; i < MODEL_TOTAL_INPUT_ELEMENTS; ++i) {
        input_buffer[i] = quantize_input(mfcc_window_norm[i]);
    }

    // 2. Inférence du modèle sur microcontrôleur
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        printf("[ERREUR] Erreur lors de l'exécution de interpreter->Invoke() !\n");
        return false;
    }

    // 3. Récupération de la sortie quantifiée et dé-quantification
    int8_t quant_output = model_output->data.int8[0];
    float logit = dequantize_output(quant_output);
    float proba = compute_probability(logit);

    if (out_proba) {
        *out_proba = proba;
    }

    return is_abnormal(proba);
}

int main() {
    printf("Démarrage du système de détection cardiaque embarqué...\n");

    if (!setup_tflite_micro()) {
        return -1;
    }

    // Simulation d'une fenêtre de test de 2,5 s (13 MFCC x 75 trames = 975 valeurs)
    float test_window[MODEL_TOTAL_INPUT_ELEMENTS] = {0.0f};

    float proba = 0.0f;
    bool anormal = classifier_fenetre_pcg(test_window, &proba);

    printf("Résultat de l'inférence :\n");
    printf("  Probabilité d'anomalie : %.3f (Seuil = %.2f)\n", proba, MODEL_DECISION_THRESHOLD);
    printf("  Verdict médical        : %s\n", anormal ? "ANORMAL (Alerte)" : "NORMAL (Sain)");

    return 0;
}
