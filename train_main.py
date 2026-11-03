
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory

# 1. Dataset location
dataset_path = os.path.join(
    "dataset",
    "traffic sign recognication",
    "crop_dataset",
    "crop_dataset"
)

# 2. Image settings
IMG_SIZE = (32, 32)
BATCH_SIZE = 32
SEED = 42

# 3. Load training and validation images
train_ds = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

# 4. Improve loading performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# 5. Build the CNN model
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(43, activation="softmax")
])

# 6. Configure model training
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 7. Display model structure
model.summary()

# 8. Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# 9. Save the trained model
os.makedirs("models", exist_ok=True)
model.save("models/traffic_sign_model.keras")

print("\nTraining completed successfully!")
print("Model saved to models/traffic_sign_model.keras")
