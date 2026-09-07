# Handwritten Digit Recognition (CNN + Tkinter)

A desktop app that lets you draw a digit (0–9) with your mouse and predicts it
using a CNN trained on MNIST.

## Files
- `train_model.py` — builds, trains, and saves the CNN (`mnist_cnn_model.h5`)
- `digit_recognizer_gui.py` — Tkinter GUI: draw, Predict, Clear
- `requirements.txt` — Python dependencies

## Setup

```bash
pip install -r requirements.txt
```

## Step 1: Train the model

```bash
python train_model.py
```

This downloads MNIST automatically (via `tf.keras.datasets.mnist`), trains a
CNN with:
- 2x Conv2D(32) → MaxPooling → Dropout(0.25)
- 2x Conv2D(64) → MaxPooling → Dropout(0.25)
- Dense(256) → Dropout(0.5)
- Dense(10, softmax)

Training uses light image augmentation (rotation/shift/zoom) so the model
generalizes better to freehand mouse drawings, which look different from
clean MNIST digits. Typical test accuracy: ~99%.

The trained model is saved to `mnist_cnn_model.h5`.

## Step 2: Run the GUI

```bash
python digit_recognizer_gui.py
```

- Draw a digit anywhere on the black canvas (white "chalk" strokes).
- Click **Predict** — the predicted digit, confidence %, and a bar chart of
  all 10 class probabilities are shown.
- Click **Clear** to reset the canvas.

## How the pipeline works (Integration)

1. Strokes drawn on the Tkinter canvas are mirrored in real time onto an
   in-memory PIL image (same size as the canvas).
2. On **Predict**:
   - The drawing is cropped to its bounding box.
   - Padded to a square and resized to 20×20 (matching how MNIST digits are
     typically framed).
   - Centered in a 28×28 black canvas (same convention as MNIST).
   - Pixel values normalized to `[0, 1]` and reshaped to `(1, 28, 28, 1)`.
   - Fed into the loaded CNN via `model.predict()`.
   - The `argmax` of the softmax output is the predicted digit; that
     probability is shown as the confidence percentage.

## Notes / Tips
- If predictions seem off, draw digits large and centered, similar to how
  you'd write on a whiteboard — thin/tiny strokes can be harder to classify.
- If you want higher accuracy on real hand-drawn input, increase the
  augmentation ranges in `train_model.py` or train longer.
