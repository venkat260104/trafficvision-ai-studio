import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.utils import load_img, img_to_array
from sklearn.metrics import accuracy_score, classification_report

BATCH_SIZE = 32

BASE_PATH = os.path.join("dataset", "traffic sign recognication")
TEST_PATH = os.path.join(BASE_PATH, "test_data", "test_data")
CSV_PATH = os.path.join(BASE_PATH, "test_labels.csv")

ORIGINAL_MODEL_PATH = "models/traffic_sign_model.keras"
COMPRESSED_MODEL_PATH = "models/traffic_sign_model_quant.tflite"

print("\nChecking required files...")

required_files = [
    TEST_PATH,
    CSV_PATH,
    ORIGINAL_MODEL_PATH,
    COMPRESSED_MODEL_PATH
]

for path in required_files:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file or folder not found: {path}"
        )

print("All required files found!")

# Load original CNN model
print("\nLoading original CNN model...")

original_model = tf.keras.models.load_model(
    ORIGINAL_MODEL_PATH
)

print("Original CNN loaded successfully!")

# Load compressed TFLite model
print("\nLoading compressed CNN model...")

interpreter = tf.lite.Interpreter(
    model_path=COMPRESSED_MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Compressed CNN loaded successfully!")

# Load test labels
print("\nLoading test labels...")

df = pd.read_csv(CSV_PATH)

print("Total CSV records:", len(df))


# Function to load images in batches
def load_test_batch(batch_df):

    images = []
    labels = []

    for _, row in batch_df.iterrows():

        image_path = os.path.join(
            TEST_PATH,
            str(row["image"])
        )

        if not os.path.exists(image_path):
            print("Missing image:", image_path)
            continue

        img = load_img(
            image_path,
            target_size=(32, 32)
        )

        img_array = img_to_array(img)

        images.append(img_array)
        labels.append(int(row["label"]))

    if not images:
        return None, None

    return (
        np.array(images, dtype=np.float32),
        np.array(labels, dtype=np.int64)
    )


# Store predictions and evaluation times
original_predictions = []
compressed_predictions = []
actual_labels = []

original_time = 0.0
compressed_time = 0.0

processed_images = 0
total_rows = len(df)

print("\nStarting full-dataset evaluation...")

# Process dataset in batches
for start in range(0, total_rows, BATCH_SIZE):

    end = min(start + BATCH_SIZE, total_rows)

    X_batch, y_batch = load_test_batch(
        df.iloc[start:end]
    )

    if X_batch is None:
        continue

    # Original CNN prediction
    start_time = time.time()

    original_probabilities = original_model.predict(
        X_batch,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    original_batch_predictions = np.argmax(
        original_probabilities,
        axis=1
    )

    original_time += time.time() - start_time

    # Compressed TFLite prediction
    start_time = time.time()

    compressed_batch_predictions = []

    for image in X_batch:

        input_tensor = np.expand_dims(
            image,
            axis=0
        )

        interpreter.set_tensor(
            input_details[0]["index"],
            input_tensor
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_details[0]["index"]
        )

        compressed_batch_predictions.append(
            int(np.argmax(output, axis=1)[0])
        )

    compressed_time += time.time() - start_time

    # Store results
    original_predictions.extend(
        original_batch_predictions.tolist()
    )

    compressed_predictions.extend(
        compressed_batch_predictions
    )

    actual_labels.extend(
        y_batch.tolist()
    )

    processed_images += len(y_batch)

    print(
        f"Processed {processed_images} / {total_rows} images"
    )


# Check evaluation
if processed_images == 0:
    raise ValueError(
        "No valid test images were processed."
    )


# Convert lists into NumPy arrays
y_test = np.array(actual_labels)

original_predictions = np.array(
    original_predictions
)

compressed_predictions = np.array(
    compressed_predictions
)


# Calculate accuracy
original_accuracy = accuracy_score(
    y_test,
    original_predictions
)

compressed_accuracy = accuracy_score(
    y_test,
    compressed_predictions
)


# Calculate prediction agreement
matching_predictions = np.sum(
    original_predictions == compressed_predictions
)

prediction_agreement = (
    matching_predictions / len(y_test)
) * 100


# Calculate model sizes
original_size = (
    os.path.getsize(ORIGINAL_MODEL_PATH) / (1024 * 1024)
)

compressed_size = (
    os.path.getsize(COMPRESSED_MODEL_PATH) / (1024 * 1024)
)

space_saved = original_size - compressed_size

reduction_percentage = (
    space_saved / original_size
) * 100


# Display evaluation results
print("\n" + "=" * 60)

print("       TRAFFIC VISION AI MODEL EVALUATION")

print("=" * 60)


print("\nORIGINAL CNN MODEL")

print(
    "Accuracy:",
    round(original_accuracy * 100, 2),
    "%"
)

print(
    "Prediction Time:",
    round(original_time, 2),
    "seconds"
)


print("\nCOMPRESSED CNN MODEL")

print(
    "Accuracy:",
    round(compressed_accuracy * 100, 2),
    "%"
)

print(
    "Prediction Time:",
    round(compressed_time, 2),
    "seconds"
)


print("\nPREDICTION COMPARISON")

print(
    "Matching Predictions:",
    matching_predictions,
    "/",
    len(y_test)
)

print(
    "Prediction Agreement:",
    round(prediction_agreement, 2),
    "%"
)


print("\nMODEL SIZE COMPARISON")

print(
    "Original Model:",
    round(original_size, 2),
    "MB"
)

print(
    "Compressed Model:",
    round(compressed_size, 2),
    "MB"
)

print(
    "Space Saved:",
    round(space_saved, 2),
    "MB"
)

print(
    "Size Reduction:",
    round(reduction_percentage, 2),
    "%"
)


# Classification reports
print("\nORIGINAL MODEL CLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        original_predictions,
        zero_division=0
    )
)


print("\nCOMPRESSED MODEL CLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        compressed_predictions,
        zero_division=0
    )
)


# Save evaluation results to CSV
results = pd.DataFrame({

    "Actual": y_test,

    "Original_Prediction": original_predictions,

    "Compressed_Prediction": compressed_predictions

})

results["Predictions_Match"] = (
    results["Original_Prediction"]
    == results["Compressed_Prediction"]
)

results.to_csv(
    "compressed_model_evaluation.csv",
    index=False
)


print("\nEvaluation results saved successfully!")

print(
    "CSV file: compressed_model_evaluation.csv"
)

print("\nFULL DATASET EVALUATION COMPLETED!")