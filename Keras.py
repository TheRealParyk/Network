import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Data
x_train_weight = tf.constant([5, 10, 15, 20, 25], dtype=tf.float32)
x_train_distance = tf.constant([50, 100, 150, 200, 250], dtype=tf.float32)
y_train_fuel = tf.constant([500, 1000, 1500, 2000, 2500], dtype=tf.float32)

# Stack features
x_train = tf.stack([x_train_weight, x_train_distance], axis=1)

# 🔹 Normalize data
normalizer = layers.Normalization()
normalizer.adapt(x_train)

# 🔹 Simpler and more precise model
model = keras.Sequential([
    normalizer,
    layers.Dense(1)  # linear model (best for this dataset)
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss='mean_squared_error',
    metrics=['mae']
)

# 🔹 Train with more stability
model.fit(x_train, y_train_fuel, epochs=300, verbose=0)

print('Training completed!')

# Test data
x_test_weight = tf.constant([8, 12, 18], dtype=tf.float32)
x_test_distance = tf.constant([60, 120, 180], dtype=tf.float32)
x_test = tf.stack([x_test_weight, x_test_distance], axis=1)

predictions = model.predict(x_test)

print("Input data:", x_test)
tf.print("Fuel:", tf.squeeze(predictions))
