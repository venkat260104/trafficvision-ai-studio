
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.utils import load_img, img_to_array
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. CONFIGURATION
# ==========================================

BATCH_SIZE = 32

BASE_PATH = os.path.join(
    "dataset", "traffic sign recognication"
)

TEST_PATH = os.path.join(
    BASE_PATH, "test_data", "test_data"
)

CSV_PATH = os.path.join(
    BASE_PATH, "test_labels.csv"
)

ORIGINAL_MODEL_PATH = (
    "models/traffic_sign_model.keras"
)

COMPRESSED_MODEL_PATH = (
    "models/traffic_sign_model_quant.tflite"
)


# ==========================================
# 2. CHECK REQUIRED FILES
# ==========================================

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


# ==========================================
# 3. LOAD ORIGINAL CNN MODEL
# ==========================================

print("\nLoading original CNN model...")

original_model = tf.keras.models.load_model(
    ORIGINAL_MODEL_PATH
)

print("Original CNN loaded successfully!")


# ==========================================
# 4. LOAD COMPRESSED TFLITE MODEL
# ==========================================

print("\nLoading compressed CNN model...")

interpreter = tf.lite.Interpreter(
    model_path=COMPRESSED_MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Compressed CNN loaded successfully!")


# ==========================================
# 5. LOAD TEST LABELS
# ==========================================

print("\nLoading test labels...")

df = pd.read_csv(CSV_PATH)

print("Total CSV records:", len(df))


# ==========================================
# 6. PREPARE TEST IMAGES
# ==========================================

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

    if len(images) == 0:
        return None, None

    X_batch = np.array(
        images,
        dtype=np.float32
    )

    y_batch = np.array(
        labels,
        dtype=np.int64
    )

    return X_batch, y_batch


# ==========================================
# 7. EVALUATE BOTH MODELS IN BATCHES
# ==========================================

original_predictions = []
compressed_predictions = []
actual_labels = []

total_rows = len(df)
processed_images = 0

original_time = 0.0
compressed_time = 0.0

print("\nStarting full-dataset evaluation...")


for start in range(0, total_rows, BATCH_SIZE):

    end = min(start + BATCH_SIZE, total_rows)

    batch_df = df.iloc[start:end]

    X_batch, y_batch = load_test_batch(batch_df)

    if X_batch is None:
        continue

    # ------------------------------
    # Original CNN prediction
    # ------------------------------

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

    # ------------------------------
    # Compressed TFLite prediction
    # ------------------------------

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

        predicted_class = int(
            np.argmax(output, axis=1)[0]
        )

        compressed_batch_predictions.append(
            predicted_class
        )

    compressed_time += time.time() - start_time

    # ------------------------------
    # Store results
    # ------------------------------

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


# ==========================================
# 8. VALIDATE EVALUATION RESULTS
# ==========================================

if processed_images == 0:
    raise ValueError(
        "No valid test images were processed."
    )

y_test = np.array(actual_labels)

original_predictions = np.array(
    original_predictions
)

compressed_predictions = np.array(
    compressed_predictions
)

print("\nDataset evaluation completed!")

print("Successfully evaluated images:", processed_images)


# ==========================================
# 9. CALCULATE ACCURACY
# ==========================================

original_accuracy = accuracy_score(
    y_test,
    original_predictions
)

compressed_accuracy = accuracy_score(
    y_test,
    compressed_predictions
)


# ==========================================
# 10. PREDICTION AGREEMENT
# ==========================================

matching_predictions = np.sum(
    original_predictions == compressed_predictions
)

prediction_agreement = (
    matching_predictions / len(y_test)
) * 100


# ==========================================
# 11. MODEL SIZE COMPARISON
# ==========================================

original_size = os.path.getsize(
    ORIGINAL_MODEL_PATH
) / (1024 * 1024)

compressed_size = os.path.getsize(
    COMPRESSED_MODEL_PATH
) / (1024 * 1024)

space_saved = original_size - compressed_size

reduction_percentage = (
    space_saved / original_size
) * 100


# ==========================================
# 12. DISPLAY RESULTS
# ==========================================

print("\n" + "=" * 60)
print("       TRAFFIC VISION AI MODEL EVALUATION")
print("=" * 60)

print("\nORIGINAL CNN MODEL")
print("-" * 60)

print(
    "Accuracy:",
    round(original_accuracy * 100, 2), "%"
)

print(
    "Prediction Time:",
    round(original_time, 2), "seconds"
)


print("\nCOMPRESSED CNN MODEL")
print("-" * 60)

print(
    "Accuracy:",
    round(compressed_accuracy * 100, 2), "%"
)

print(
    "Prediction Time:",
    round(compressed_time, 2), "seconds"
)


print("\nPREDICTION COMPARISON")
print("-" * 60)

print(
    "Matching Predictions:",
    matching_predictions,
    "/",
    len(y_test)
)

print(
    "Prediction Agreement:",
    round(prediction_agreement, 2), "%"
)


print("\nMODEL SIZE COMPARISON")
print("-" * 60)

print(
    "Original Model:",
    round(original_size, 2), "MB"
)

print(
    "Compressed Model:",
    round(compressed_size, 2), "MB"
)

print(
    "Space Saved:",
    round(space_saved, 2), "MB"
)

print(
    "Size Reduction:",
    round(reduction_percentage, 2), "%"
)


# ==========================================
# 13. CLASSIFICATION REPORTS
# ==========================================

print("\nORIGINAL MODEL CLASSIFICATION REPORT")
print("-" * 60)

print(
    classification_report(
        y_test,
        original_predictions,
        zero_division=0
    )
)


print("\nCOMPRESSED MODEL CLASSIFICATION REPORT")
print("-" * 60)

print(
    classification_report(
        y_test,
        compressed_predictions,
        zero_division=0
    )
)


# ==========================================
# 14. SAVE EVALUATION RESULTS
# ==========================================

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

print("CSV file: compressed_model_evaluation.csv")


# ==========================================
# 15. FINAL STATUS
# ==========================================

print("\n" + "=" * 60)
print("FULL DATASET EVALUATION COMPLETED!")
print("=" * 60)
