import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Reshape to (N, 28, 28, 1) and normalize to [0, 1]
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    return (x_train, y_train), (x_test, y_test)


def build_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("Loading MNIST data...")
    (x_train, y_train), (x_test, y_test) = load_data()

    print("Building model...")
    model = build_model()
    model.summary()

    # Light augmentation helps generalize to hand-drawn digits
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )
    datagen.fit(x_train)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    print("Training...")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        epochs=20,
        validation_data=(x_test, y_test),
        callbacks=[early_stop],
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss:     {test_loss:.4f}")

    # Save the trained model (architecture + weights)
    model.save("mnist_cnn_model.h5")
    print("Model saved to mnist_cnn_model.h5")

    # Optional: plot training curves
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(history.history["accuracy"], label="train")
        axes[0].plot(history.history["val_accuracy"], label="val")
        axes[0].set_title("Accuracy")
        axes[0].legend()

        axes[1].plot(history.history["loss"], label="train")
        axes[1].plot(history.history["val_loss"], label="val")
        axes[1].set_title("Loss")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig("training_curves.png")
        print("Training curves saved to training_curves.png")
    except Exception as e:
        print(f"Could not save plot: {e}")


if __name__ == "__main__":
    main()
