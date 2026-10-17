// =============================================================================
// Project: TinyML Waste Classification on ESP32 using TensorFlow Lite Micro
// Author: Hari Kruthik SK (23BEC0358)
// Target: ESP32 (Xtensa LX6 / ESP32-S3)
// Model: Quantized INT8 MobileNetV3Large (2.19 MB)
// =============================================================================

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Classes: 0:Cardboard, 1:Glass, 2:Metal, 3:Paper, 4:Plastic, 5:Trash
const char* CLASS_NAMES[] = {
    "Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"
};

// Memory Arena for TFLite Micro Allocator
constexpr int kTensorArenaSize = 120 * 1024; // 120 KB SRAM buffer
uint8_t tensor_arena[kTensorArenaSize];

// External reference to the compiled C-byte array model
extern const unsigned char g_waste_model_data[];
extern const int g_waste_model_data_len;

tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n--- ESP32 TinyML Waste Classifier Initializing ---");

    // 1. Map Model Schema
    model = tflite::GetModel(g_waste_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("[ERROR] Model schema version mismatch!");
        while (1);
    }

    // 2. Resolve Operations
    static tflite::AllOpsResolver resolver;

    // 3. Build MicroInterpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter
    );
    interpreter = &static_interpreter;

    // 4. Allocate Tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println("[ERROR] AllocateTensors() failed! Check tensor arena size.");
        while (1);
    }

    // 5. Query Input and Output Pointers
    input = interpreter->input(0);
    output = interpreter->output(0);

    Serial.printf("[OK] Interpreter ready. Input dims: %d x %d x %d\n",
                  input->dims->data[1], input->dims->data[2], input->dims->data[3]);
}

void loop() {
    // Check for serial frame or triggered capture
    if (Serial.available()) {
        char cmd = Serial.read();
        if (cmd == 'I') { // Run inference command
            unsigned long start_us = micros();

            // Run quantized INT8 inference
            TfLiteStatus invoke_status = interpreter->Invoke();
            unsigned long duration_ms = (micros() - start_us) / 1000;

            if (invoke_status != kTfLiteOk) {
                Serial.println("[ERROR] Inference invocation failed!");
                return;
            }

            // Dequantize and find argmax class
            int top_class = 0;
            int8_t max_val = -128;
            for (int i = 0; i < 6; i++) {
                int8_t val = output->data.int8[i];
                if (val > max_val) {
                    max_val = val;
                    top_class = i;
                }
            }

            // Output result over serial
            Serial.printf("[RESULT] Category: %s | Score: %d | Latency: %lu ms\n",
                          CLASS_NAMES[top_class], max_val, duration_ms);
        }
    }
}
