import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import tensorflow as tf

MODEL_PATH = "mnist_cnn_model.h5"
CANVAS_SIZE = 280          # on-screen canvas (28 * 10, for a comfortable draw area)
BRUSH_RADIUS = 8


class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")
        self.root.resizable(False, False)

        # Load the trained model
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            messagebox.showerror(
                "Model not found",
                f"Could not load '{MODEL_PATH}'.\n"
                f"Run train_model.py first to create it.\n\nDetails: {e}",
            )
            self.model = None

        # ---- Layout ----
        main_frame = tk.Frame(root, padx=10, pady=10)
        main_frame.pack()

        title = tk.Label(main_frame, text="Draw a digit (0-9)", font=("Helvetica", 14, "bold"))
        title.pack(pady=(0, 8))

        # Drawing canvas (white background, black pen -- like paper)
        self.canvas = tk.Canvas(
            main_frame, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black", cursor="cross"
        )
        self.canvas.pack()

        # PIL image mirrors the canvas so we can process it as pixel data
        self.pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)  # black bg
        self.pil_draw = ImageDraw.Draw(self.pil_image)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last_point)
        self.last_x, self.last_y = None, None

        # Buttons
        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(pady=10)

        predict_btn = tk.Button(
            btn_frame, text="Predict", width=12, command=self.predict, bg="#4CAF50", fg="white"
        )
        predict_btn.grid(row=0, column=0, padx=5)

        clear_btn = tk.Button(
            btn_frame, text="Clear", width=12, command=self.clear_canvas, bg="#f44336", fg="white"
        )
        clear_btn.grid(row=0, column=1, padx=5)

        # Result display
        self.result_label = tk.Label(
            main_frame, text="Prediction: -", font=("Helvetica", 20, "bold")
        )
        self.result_label.pack(pady=(10, 0))

        self.confidence_label = tk.Label(
            main_frame, text="Confidence: -", font=("Helvetica", 12)
        )
        self.confidence_label.pack()

        # Bar of per-class probabilities
        self.prob_canvas = tk.Canvas(main_frame, width=CANVAS_SIZE, height=120, bg="white")
        self.prob_canvas.pack(pady=(10, 0))

    # ---------- Drawing ----------
    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            # Draw on the visible Tkinter canvas
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                width=BRUSH_RADIUS * 2, fill="white",
                capstyle=tk.ROUND, smooth=True,
            )
            # Mirror the stroke onto the PIL image used for prediction
            self.pil_draw.line(
                [self.last_x, self.last_y, x, y],
                fill=255, width=BRUSH_RADIUS * 2,
            )
        else:
            # Single dot (click without drag)
            r = BRUSH_RADIUS
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
            self.pil_draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

        self.last_x, self.last_y = x, y

    def reset_last_point(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        self.result_label.config(text="Prediction: -")
        self.confidence_label.config(text="Confidence: -")
        self.prob_canvas.delete("all")

    # ---------- Preprocessing + Prediction ----------
    def preprocess(self):
        """
        Convert the drawn PIL image (280x280, white digit on black bg)
        into a normalized 28x28x1 array matching MNIST's format.
        """
        img = self.pil_image.copy()

        # Crop to the bounding box of the drawing so the digit is centered
        bbox = img.getbbox()
        if bbox is not None:
            img = img.crop(bbox)

        # Pad to a square with some margin, then resize to 20x20
        # (MNIST digits are typically ~20px tall within a 28px frame)
        w, h = img.size
        size = max(w, h)
        padded = Image.new("L", (size, size), color=0)
        padded.paste(img, ((size - w) // 2, (size - h) // 2))

        img20 = padded.resize((20, 20), Image.LANCZOS)

        # Place the 20x20 digit into a 28x28 black canvas, centered
        final = Image.new("L", (28, 28), color=0)
        final.paste(img20, (4, 4))

        arr = np.array(final).astype("float32") / 255.0
        arr = arr.reshape(1, 28, 28, 1)
        return arr, final

    def predict(self):
        if self.model is None:
            messagebox.showwarning("No model", "Model is not loaded.")
            return

        if self.pil_image.getbbox() is None:
            messagebox.showinfo("Empty canvas", "Please draw a digit first.")
            return

        x, _ = self.preprocess()
        probs = self.model.predict(x, verbose=0)[0]
        digit = int(np.argmax(probs))
        confidence = float(probs[digit]) * 100

        self.result_label.config(text=f"Prediction: {digit}")
        self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")
        self.draw_probabilities(probs)

    def draw_probabilities(self, probs):
        self.prob_canvas.delete("all")
        n = len(probs)
        bar_width = CANVAS_SIZE / n
        max_height = 90

        for i, p in enumerate(probs):
            bar_h = p * max_height
            x0 = i * bar_width + 4
            x1 = (i + 1) * bar_width - 4
            y1 = max_height + 10
            y0 = y1 - bar_h
            color = "#4CAF50" if i == int(np.argmax(probs)) else "#90CAF9"
            self.prob_canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")
            self.prob_canvas.create_text(
                (x0 + x1) / 2, y1 + 10, text=str(i), font=("Helvetica", 9)
            )


def main():
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
