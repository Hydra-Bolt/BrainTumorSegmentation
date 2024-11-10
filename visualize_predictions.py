import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import imageio
import tifffile as tiff
import tensorflow as tf
import os
from functools import lru_cache

# function that takes both image & mask path and return the image
def process_path(image_path, mask_path):
    img = tiff.imread(image_path)
    img = img[:, :, 0:3]
    img = tf.image.convert_image_dtype(img, tf.float32)

    mask = tiff.imread(mask_path)
    # Check if mask is 2D or 3D
    if mask.ndim == 2:
        mask = np.expand_dims(mask, axis=-1)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    return img, mask

def preprocess(image, mask):
    input_image = tf.image.resize(image, (128, 128), method='nearest')
    input_mask = tf.image.resize(mask, (128, 128), method='nearest')

    return input_image, input_mask

def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def predict(model, image):
    pred_mask = model.predict(image[tf.newaxis, ...])
    pred_mask = create_mask(pred_mask)
    return pred_mask

def create_gif_with_predictions(images, masks, model, gif_path='mri_with_predictions.gif', duration=0.5):
    frames = []
    print(images)
    for i in range(len(images)):
        image_path = images[i]
        mask_path = masks[i]
        
        image, mask = process_path(image_path, mask_path)
        image, mask = preprocess(image, mask)
        pred_mask = predict(model, image)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        ax1.imshow(image, cmap='gray')
        ax1.imshow(mask, cmap='jet', alpha=0.5)
        ax1.set_title('MRI Image with True Mask Overlay')
        ax1.axis('off')
        
        ax2.imshow(image, cmap='gray')
        ax2.imshow(pred_mask, cmap='jet', alpha=0.5)
        ax2.set_title('MRI Image with Predicted Mask Overlay')
        ax2.axis('off')
        
        fig.canvas.draw()

        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(gif_path, frames, duration=duration)

def visualize_predictions(model, data_dir, save_dir, num_images=5):
    patients = glob(f"{data_dir}/*")
    # List all tif files in the directory
    for patient in patients[:num_images]:
        mris_with_mask = glob(f"{patient}/*.tif")
        # Separate mask and image files
        image_files = [f for f in mris_with_mask if '_mask' not in f]
        mask_files = [f for f in mris_with_mask if '_mask' in f]
        # Sort the files
        images = sorted(image_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        masks = sorted(mask_files, key=lambda x: int(x.split('_')[-2].split('.')[-1]))
        
        # Create a gif with predictions for the first num_images
        create_gif_with_predictions(images, masks, model, gif_path=os.path.join(save_dir, f"{patient}.gif"))

@lru_cache(maxsize=None)
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

if __name__ == "__main__":
    model = load_model('model/BrainMRISegment-Predictor.keras')
    visualize_predictions(model, 'dataset/kaggle_3m', 'output', num_images=30)