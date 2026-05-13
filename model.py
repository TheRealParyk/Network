from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

DATASET_PATH = 'dataset/'
IMG_SIZE = (128, 128)
BATCH_SIZE = 8

# Ensure dataset exists
if not os.path.exists(DATASET_PATH):
    raise ValueError(f"Dataset path not found: {DATASET_PATH}")

# 🔹 Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

class_names = train_ds.class_names
num_classes = len(class_names)

# 🔹 Normalize
normalizer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalizer(x), y))

# 🔹 Performance optimization
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# 🔹 Model
model = keras.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',  # FIXED
    metrics=['accuracy']
)

# 🔹 Train
model.fit(train_ds, epochs=10)

# Save model (modern format)
model.save("image_classifier.keras")

print("Training completed!")

# ------------------- Prediction -------------------

# Load model once
model = tf.keras.models.load_model("image_classifier.keras")

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File not found at path: {image_path}")
        return

    try:
        img = Image.open(image_path)
        img.verify()
        img = Image.open(image_path)
    except (OSError, IOError):
        print(f"Error: Corrupted image - {image_path}")
        return

    # 🔹 Preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # FIXED COLOR
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_class = class_names[predicted_index]

    print(f"The model has determined: {predicted_class}")

    plt.imshow(img[0])
    plt.title(f"The model has determined: {predicted_class}")
    plt.axis('off')
    plt.show()


# Example usage
predict_image("dataset/dogs/dog 8.jpg")
