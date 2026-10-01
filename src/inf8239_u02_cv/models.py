from __future__ import annotations


def build_dense(tf):
    """Build the dense baseline for normalized Fashion-MNIST images."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input((28, 28, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ],
        name="dense_baseline",
    )


def build_cnn(tf):
    """Build the compact CNN used in the exercise."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input((28, 28, 1)),
            tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(10, activation="softmax"),
        ],
        name="fashion_mnist_cnn",
    )