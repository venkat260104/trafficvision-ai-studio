
import tensorflow as tf
import os

# Original trained CNN model
model_path = "models/traffic_sign_model.keras"

print("Loading original CNN model...")

model = tf.keras.models.load_model(model_path)

# Create TensorFlow Lite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable model optimization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("Compressing CNN model...")

# Convert model
tflite_model = converter.convert()

# Save compressed model separately
output_path = "models/traffic_sign_model_quant.tflite"

with open(output_path, "wb") as f:
    f.write(tflite_model)

# Compare file sizes
original_size = os.path.getsize(model_path) / (1024 * 1024)
compressed_size = os.path.getsize(output_path) / (1024 * 1024)

print("\nCompression completed!")
print(f"Original model: {original_size:.2f} MB")
print(f"Compressed model: {compressed_size:.2f} MB")
print(f"Space saved: {original_size - compressed_size:.2f} MB")
