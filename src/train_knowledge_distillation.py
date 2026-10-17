# =============================================================================
# Knowledge Distillation & INT8 Quantization for TinyML Waste Classification
# Author: Hari Kruthik SK (23BEC0358)
# Target: ESP32 (Xtensa LX6 / ESP32-S3)
# Teacher: EfficientNetB0 | Student: MobileNetV3Large (alpha=0.75)
# Dataset: TrashNet (6 Classes, 2527 images)
# =============================================================================

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -----------------------------------------------------------------------------
# 1. Dataset Configuration & Preprocessing
# -----------------------------------------------------------------------------
BATCH_SIZE = 32
IMG_SIZE = (128, 128) # Downsampled for MCU SRAM footprint
NUM_CLASSES = 6
CLASSES = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

# Class distribution:
# Cardboard: 403 | Glass: 501 | Metal: 410 | Paper: 594 | Plastic: 482 | Trash: 137

def build_teacher_model():
    """Build high-capacity teacher network using EfficientNetB0."""
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = True

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, name='logits')(x)
    return keras.Model(inputs, outputs, name="Teacher_EfficientNetB0")

def build_student_model():
    """Build compact student network using MobileNetV3Large (alpha=0.75)."""
    base_model = tf.keras.applications.MobileNetV3Large(
        input_shape=(*IMG_SIZE, 3),
        alpha=0.75,
        minimalistic=False,
        include_top=False,
        weights='imagenet'
    )
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
    x = base_model(x, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, name='logits')(x)
    return keras.Model(inputs, outputs, name="Student_MobileNetV3Large")

# -----------------------------------------------------------------------------
# 2. Knowledge Distillation Trainer Class
# -----------------------------------------------------------------------------
class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha=0.1, temperature=3.0):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        x, y = data
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            student_predictions = self.student(x, training=True)
            # Standard hard label cross-entropy loss
            student_loss = self.student_loss_fn(y, student_predictions)
            # Soft label Kullback-Leibler divergence distillation loss
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            ) * (self.temperature ** 2)

            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, student_predictions)

        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss, "distillation_loss": distillation_loss})
        return results

# -----------------------------------------------------------------------------
# 3. Post-Training Quantization (INT8)
# -----------------------------------------------------------------------------
def quantize_to_tflite_micro(student_model, representative_data_gen, output_path="waste_model_int8.tflite"):
    """Quantize trained student model to full INT8 for ESP32 TFLite Micro deployment."""
    converter = tf.lite.TFLiteConverter.from_keras_model(student_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Enforce integer-only quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_quant_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_quant_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[OK] Quantized INT8 model successfully saved: {output_path} ({size_mb:.2f} MB)")
    return output_path

if __name__ == "__main__":
    print("[INFO] Initializing Teacher-Student Knowledge Distillation Pipeline...")
    teacher = build_teacher_model()
    student = build_student_model()
    print("[OK] Models constructed successfully.")
