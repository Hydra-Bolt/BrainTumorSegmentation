import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import imageio
import tifffile as tiff
import tensorflow as tf
import os
from functools import lru_cache


class BrainTumorSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Tumor Segmentation")
        self.root.geometry("1000x800")
        self.root.configure(bg="#323232")
        
        self.main_frame = tk.Frame(root, bg="#323232", highlightbackground="#C3D825")
        self.main_frame.pack(pady=20, padx=20)
        
        self.label = tk.Label(self.main_frame, text="Brain Tumor Segmentation Tool", font=("Cartograph CF", 40), bg="#C3D825", fg="#323232")
        self.label.grid(row=0, column=0, columnspan=2, pady=10)
        
        self.upload_button = tk.Button(self.main_frame, text="Upload MRI Image", command=self.upload_image, font=("Cartograph CF", 13), bg="#323232", fg="#C3D825")
        self.upload_button.grid(row=1, column=0, pady=10)
        
        self.segment_button = tk.Button(self.main_frame, text="Segment Tumor", command=self.segment_tumor, state=tk.DISABLED, font=("Cartograph CF", 13), bg="#323232", fg="#C3D825")
        self.segment_button.grid(row=1, column=1, pady=10)
        
        self.image_label = tk.Label(self.main_frame, bg="#323232")
        self.image_label.grid(row=2, column=0, columnspan=2, pady=20)
        
        self.result_label = tk.Label(self.main_frame, text="", font=("Cartograph CF", 13), bg="#323232", fg="#C3D825")
        self.result_label.grid(row=3, column=0, columnspan=2, pady=10)
        
        self.image_path = None

    def upload_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if self.image_path:
            image = Image.open(self.image_path)
            image = image.resize((300, 300))
            self.image = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.image)
            self.segment_button.config(state=tk.NORMAL)
    
    @lru_cache(maxsize=None)
    def load_model(self, model_path):
        return tf.keras.models.load_model(model_path)
    
    def segment_tumor(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please upload an MRI image first.")
            return
        # Detect and classify the tumor first
        classification_model = self.load_model("model/BTC.keras")
        img = tf.keras.utils.load_img(self.image_path, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = classification_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
        classification_result = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
            CLASSES[np.argmax(score)], 100 * np.max(score)
        )
        self.result_label.config(text=classification_result)
        
        # Skip segmentation if the image belongs to "notumor"
        if CLASSES[np.argmax(score)] == "notumor":
            return
        
        # Segment the tumor
        segmentation_model = self.load_model("model/BrainMRISegment-Predictor.keras")
        
        if self.image_path.lower().endswith(('.tif', '.tiff')):
            image = tiff.imread(self.image_path)
            image = image[:, :, 0:3]
        else:
            image = imageio.imread(self.image_path)
            if image.ndim == 2:  # Grayscale image
                image = np.stack((image,) * 3, axis=-1)
            elif image.shape[2] == 4:  # RGBA image
                image = image[:, :, :3]
        
        image = tf.image.convert_image_dtype(image, tf.float32)
        input_image = tf.image.resize(image, (128, 128), method='nearest')
        input_image = input_image[tf.newaxis, ...]

        pred_mask = segmentation_model.predict(input_image)
        pred_mask = tf.math.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[0]

        # Create the overlay
        image = Image.open(self.image_path)
        image = image.resize((128, 128))
        image = image.convert("RGBA")

        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_data = overlay.load()

        for i in range(128):
            for j in range(128):
                if pred_mask[i, j] == 1:
                    overlay_data[j, i] = (255, 0, 0, 255)

        image = Image.alpha_composite(image, overlay)
        image = image.resize((300, 300))
        
        self.image = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.image)


        
if __name__ == "__main__":
    root = tk.Tk()
    app = BrainTumorSegmentationApp(root)
    root.mainloop()