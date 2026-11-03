
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.metrics import accuracy_score, classification_report

# Load the trained model
model = tf.keras.models.load_model(
    "models/traffic_sign_model.keras"
)

# Dataset paths
base_path = os.path.join(
    "dataset", "traffic sign recognication"
)
test_path = os.path.join(
    base_path, "test_data", "test_data"
)
csv_path = os.path.join(base_path, "test_labels.csv")

# Load actual labels
df = pd.read_csv(csv_path)

images = []
actual_labels = []

print("Loading test images...")

for _, row in df.iterrows():
    image_path = os.path.join(test_path, row["image"])

    img = load_img(image_path, target_size=(32, 32))
    img_array = img_to_array(img)

    images.append(img_array)
    actual_labels.append(int(row["label"]))

# Convert images to NumPy array
X_test = np.array(images, dtype=np.float32)
y_test = np.array(actual_labels)

print("Total test images:", len(X_test))
print("Predicting traffic signs...")

# The model already includes image normalization
probabilities = model.predict(X_test, batch_size=32, verbose=1)
predicted_labels = np.argmax(probabilities, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, predicted_labels)

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, predicted_labels, zero_division=0))
