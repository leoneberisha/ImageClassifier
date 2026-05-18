import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions,
    MobileNetV2
)
from tensorflow.keras.preprocessing import image
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Load model një herë
model = MobileNetV2(weights="imagenet")

def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)

    predictions = model.predict(img_preprocessed)
    return decode_predictions(predictions, top=3)[0]

def open_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        return

    # Show image
    img = Image.open(file_path)
    img = img.resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)

    img_label.config(image=img_tk)
    img_label.image = img_tk

    # Predict
    results = classify_image(file_path)

    text = "📷 Rezultatet:\n\n"
    for (_, label, prob) in results:
        text += f"{label}: {round(prob * 100, 2)}%\n"

    result_label.config(text=text)

# GUI window
root = tk.Tk()
root.title("Image Classifier AI")
root.geometry("400x500")

btn = tk.Button(root, text="Zgjedh Fotografinë", command=open_image)
btn.pack(pady=10)

img_label = tk.Label(root)
img_label.pack()

result_label = tk.Label(root, text="", justify="left", font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()