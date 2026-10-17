# TinyML Garbage & Waste Classification on ESP32
### Edge Machine Learning via Teacher-Student Knowledge Distillation and INT8 Quantization

[![Target: ESP32](https://img.shields.io/badge/Target-ESP32%20Xtensa%20LX6-blue.svg)](https://www.espressif.com/)
[![Framework: TensorFlow Lite Micro](https://img.shields.io/badge/Framework-TFLite%20Micro-orange.svg)](https://www.tensorflow.org/lite/microcontrollers)
[![Model: Knowledge Distillation](https://img.shields.io/badge/Architecture-EfficientNetB0%20%E2%86%92%20MobileNetV3--0.75-green.svg)](src/train_knowledge_distillation.py)
[![Quantization: Full INT8](https://img.shields.io/badge/Quantization-Full%20INT8%20(2.19%20MB)-purple.svg)](src/train_knowledge_distillation.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-darkgreen.svg)](LICENSE)

An ultra-compact edge-computing visual garbage classification pipeline deployed onto resource-constrained microcontroller hardware (**ESP32**, 240 MHz dual-core Xtensa LX6, 520 KB SRAM). 

To circumvent the computational and memory limits of running vision convolutional neural networks on microcontrollers, this project develops a **Teacher-Student Knowledge Distillation** architecture (EfficientNetB0 Teacher distilling into a lightweight MobileNetV3Large $\alpha=0.75$ Student) coupled with representative **INT8 post-training quantization**, achieving **84.49% classification accuracy** in a **2.19 MB footprint** (+4.49 percentage-point improvement over the baseline MobileNet trained without distillation).

---

## 1. Problem Statement & Design Objectives

Automated municipal and industrial waste segregation stations typically rely on high-power edge computers ($>\$200$) or continuous wireless cloud streaming. In decentralized municipal sorting facilities, cloud bandwidth and network latency are prohibitive. Deploying local inference on low-cost microcontrollers ($<\$5$) eliminates recurring network operational costs, but imposes stringent hardware constraints:
1. **Flash Footprint Budget**: Total binary must remain below standard 4 MB SPI flash partitions ($< 2.5\text{ MB}$ allocation for model weights).
2. **RAM Allocation**: Peak tensor activations must fit within internal SRAM ($< 150\text{ KB}$ tensor arena).
3. **Integer-Only Execution**: Software floating-point emulation degrades frame rates; full INT8 quantization is mandatory for hardware integer arithmetic.

---

## 2. Dataset Distribution (TrashNet Benchmark)

The model is trained and validated on the standardized 6-class **TrashNet** benchmark consisting of **2,527 calibrated images**:

| Category | Image Count | Physical Domain Description |
| :--- | :---: | :--- |
| **Cardboard** | 403 | Corrugated cartons, shipping boxes, packaging inserts |
| **Glass** | 501 | Clear, green, and amber beverage bottles and food containers |
| **Metal** | 410 | Aluminum soda cans, tin food cans, foil containers |
| **Paper** | 594 | Office paper, newsprint, magazine sheets |
| **Plastic** | 482 | PET drink bottles, HDPE detergent containers |
| **Trash** | 137 | Contaminated miscellaneous unrecyclable solid waste |
| **Total** | **2,527** | Split: 70% Training, 15% Validation, 15% Held-out Test |

---

## 3. Knowledge Distillation Methodology

Rather than training a miniature CNN directly on hard one-hot labels, the student network is trained simultaneously against ground-truth labels and the softened output probability distribution of a high-capacity teacher network.

```
                          Input Image [224x224x3]
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
            EfficientNetB0                    MobileNetV3Large
            (TEACHER, FP32)                   (STUDENT, alpha=0.75)
                     │                               │
                     ▼                               ▼
             Soft Probs (T=3)                Soft Probs (T=3)
                     │                               │
                     └───────────────┬───────────────┘
                                     ▼
                         Distillation Loss (KL Div)
                                     +
                         Hard Classification Loss (CE)
                                     ▼
                         Backpropagation into Student
```

### Mathematical Loss Formulation
$$L_{\text{total}} = \alpha \cdot L_{\text{CE}}(y,\, \sigma(z_s)) + (1 - \alpha) \cdot T^2 \cdot L_{\text{KL}}\left(\sigma\left(\frac{z_t}{T}\right),\, \sigma\left(\frac{z_s}{T}\right)\right)$$

Where:
- $z_t, z_s$: Raw logit vectors of teacher and student networks
- $\sigma(\cdot)$: Softmax function
- $T = 3.0$: Distillation temperature parameter softening probability distributions
- $\alpha = 0.1$: Loss weighting parameter emphasizing dark knowledge transfer
- $L_{\text{KL}}$: Kullback-Leibler divergence measuring output distribution divergence:
  $$L_{\text{KL}}(P, Q) = \sum_i P(i) \cdot \log\left(\frac{P(i)}{Q(i)}\right)$$

---

## 4. Quantization & Performance Metrics

Representative dataset calibration maps 32-bit floating-point weights and activations to 8-bit signed integers:

$$q = \text{round}\left(\frac{r}{S}\right) + Z$$

| Model Architecture | Precision | Model Size | Top-1 Test Accuracy | ESP32 Flash Feasibility |
| :--- | :---: | :---: | :---: | :---: |
| **Teacher: EfficientNetB0** | FP32 | 16.2 MB | 88.12% | Infeasible (> 4 MB Flash) |
| **Student: MobileNetV3 (No Distillation)**| FP32 | 8.76 MB | 80.00% | Infeasible |
| **Student: MobileNetV3 (With Distillation)**| FP32 | 8.76 MB | 84.72% | Infeasible |
| **Quantized Student (Proposed)** | **INT8** | **2.19 MB** | **84.49%** | **Optimal (< 3 MB Flash)** |

- **Quantization Accuracy Drop**: Only **$0.23\%$** accuracy loss from FP32 to INT8.
- **Model Compression Ratio**: **$3.99\times$** reduction in weight storage.
- **Tensor Arena RAM Budget**: **120 KB SRAM** allocated on ESP32.
- **MCU Inference Latency**: ~**480 ms** on ESP32 @ 240 MHz.

---

## 5. Repository Directory Layout

```
TinyML-Garbage-Classification-ESP32/
├── README.md                           # Master technical documentation
├── .gitignore                          # Python and binary build exclusions
├── src/
│   ├── train_knowledge_distillation.py # Training, KD pipeline & INT8 export
│   └── waste_classifier.py             # Host webcam test harness
├── tools/
│   └── evaluate_tflite_model.py        # TFLite model profiling & quantization check
└── esp32_tflite_micro/                 # Microcontroller embedded deployment
    ├── esp32_waste_inference.ino       # TFLite Micro Arduino C++ sketch
    ├── model_data.h                    # Model array external header
    ├── model_data.cpp                  # Model byte alignment buffer
    └── platformio.ini                  # PlatformIO build environment
```

---

## 6. Execution & Deployment Guide

### 1. Training & Exporting the Distilled Model
```bash
python src/train_knowledge_distillation.py
```
This script trains the teacher/student models, applies symmetric INT8 calibration, and outputs `student_quantized_int8.tflite`.

### 2. Converting TFLite to C Header
```bash
xxd -i student_quantized_int8.tflite > esp32_tflite_micro/model_data.cpp
```

### 3. Deploying to ESP32 (PlatformIO)
```bash
cd esp32_tflite_micro
pio run -t upload
pio device monitor -b 115200
```

---

## 7. Authors & License

- **Lead Engineer**: Hari Kruthik SK (Reg. 23BEC0358)  
- **Affiliation**: School of Electronics Engineering (SENSE), Vellore Institute of Technology (VIT), Vellore  
- **License**: Released under the [MIT License](LICENSE).
