# TinyML Garbage & Waste Classification on ESP32

[![Target: ESP32](https://img.shields.io/badge/Target-ESP32%20Xtensa%20LX6-blue.svg)](https://www.espressif.com/)
[![Framework: TensorFlow Lite Micro](https://img.shields.io/badge/Framework-TFLite%20Micro-orange.svg)](https://www.tensorflow.org/lite/microcontrollers)
[![Model: Knowledge Distillation](https://img.shields.io/badge/Architecture-EfficientNetB0%20%E2%86%92%20MobileNetV3--0.75-green.svg)]()
[![Quantization: INT8](https://img.shields.io/badge/Quantization-Full%20INT8%20(2.19%20MB)-purple.svg)]()

An edge-computing visual garbage classification pipeline deployed onto resource-constrained microcontroller hardware (**ESP32**, 240 MHz dual-core Xtensa LX6, 520 KB SRAM). 

To circumvent the computational and memory limits of running vision CNNs on microcontrollers, this project develops a **Teacher-Student Knowledge Distillation** architecture (EfficientNetB0 Teacher distilling into a lightweight MobileNetV3Large $\alpha=0.75$ Student) coupled with representative **INT8 post-training quantization**, achieving **84.49% accuracy** in a **2.19 MB footprint** (+4.49 percentage-point improvement over the baseline MobileNet).

---

## 1. Problem Statement & Motivation

Industrial automated waste sorting systems typically rely on high-power edge PCs or continuous cloud streaming. In rural, localized, or distributed municipal deployments, cloud bandwidth and latency are prohibitive. Deploying inference locally on low-cost microcontrollers ($<\$5$) eliminates recurring network costs, but requires:
1. Strict model memory footprints below typical ESP32 flash constraints ($<4\text{ MB}$).
2. Low peak RAM allocation ($<150\text{ KB}$ tensor arena).
3. Integer-only arithmetic execution (avoiding slow software floating-point emulation).

---

## 2. Dataset Distribution (TrashNet)

Trained on the standardized 6-class **TrashNet** benchmark consisting of **2,527 images**:

| Class | Number of Images | Description |
| :--- | :---: | :--- |
| **Cardboard** | 403 | Corrugated cartons, shipping boxes, packaging inserts |
| **Glass** | 501 | Clear, green, and amber beverage bottles and jars |
| **Metal** | 410 | Aluminum soda cans, tin food cans, foil containers |
| **Paper** | 594 | Office paper, newsprint, magazine sheets |
| **Plastic** | 482 | PET drink bottles, HDPE detergent containers |
| **Trash** | 137 | Contaminated miscellaneous unrecyclable solid waste |
| **Total** | **2,527** | Split: 70% Train, 15% Validation, 15% Held-out Test |

---

## 3. Knowledge Distillation Methodology

Rather than training a miniature CNN on hard one-hot labels, the student learns from both ground-truth labels and the softened output probability distribution of a high-capacity teacher network.

```
                         Input Image
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
             EfficientNetB0      MobileNetV3Large
             (TEACHER, FP32)     (STUDENT, alpha=0.75)
                    │                   │
                    ▼                   ▼
             Soft Probs (T=3)    Soft Probs (T=3)
                    │                   │
                    └─────────┬─────────┘
                              ▼
                      Distillation Loss (KL Div)
                              +
                      Hard Classification Loss (CE)
                              ▼
                      Backpropagation into Student
```

### Loss Formulation
$$L_{\text{total}} = \alpha \cdot L_{\text{CE}}(y, \sigma(z_s)) + (1 - \alpha) \cdot T^2 \cdot L_{\text{KL}}\left(\sigma\left(\frac{z_t}{T}\right), \sigma\left(\frac{z_s}{T}\right)\right)$$
Where:
- $z_t, z_s$: Logits of teacher and student
- $T = 3.0$: Distillation temperature parameter
- $\alpha = 0.1$: Loss weighting parameter

---

## 4. Quantization & Results

Post-training quantization (PTQ) was executed with representative calibration samples. Key technical hurdles resolved included:
- Correct calibration batch sizing to prevent dynamic range clipping.
- Removing duplicate softmax layers before integer conversion.
- Compensating for class imbalance in Keras 3.

### Empirical Benchmarks

| Metric | Vanilla MobileNet Baseline | Distilled INT8 Student (Ours) | Delta |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | $80.00\%$ | **$84.49\%$** | **$+4.49\text{ pp}$** |
| **Model Size** | $12.4\text{ MB}$ (FP32) | **$2.19\text{ MB}$** (INT8) | **$5.66\times\text{ Compression}$** |
| **Tensor Arena** | $>400\text{ KB}$ | **$120\text{ KB}$** | Fits within internal SRAM |
| **Target Platform** | Cloud / GPU | **ESP32 MCU** | Fully autonomous edge |

---

## 5. Repository File Structure

```text
TinyML-Garbage-Classification-ESP32/
├── README.md
├── .gitignore
│
├── src/
│   ├── train_knowledge_distillation.py    # Teacher-Student distillation & INT8 export
│   └── waste_classifier.py                # Real-time webcam serial classifier test harness
│
└── esp32_tflite_micro/
    └── esp32_waste_inference.ino          # Native C++ TFLite Micro MCU runtime firmware
```

---

## 6. How to Deploy

### 1. Train & Export
```bash
python src/train_knowledge_distillation.py
# Exports waste_model_int8.tflite
```

### 2. Convert to C Byte Array
```bash
xxd -i waste_model_int8.tflite > esp32_tflite_micro/waste_model_data.cc
```

### 3. Flash to ESP32
Open [`esp32_tflite_micro/esp32_waste_inference.ino`](esp32_tflite_micro/esp32_waste_inference.ino) in Arduino IDE with the `TensorFlowLite_ESP32` library installed and flash to your device.
