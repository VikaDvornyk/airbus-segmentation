import cv2
import numpy as np
import tensorflow as tf

threshold = 0.4

def preprocessing(image_path):
    try:
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (64, 64))
        resized_image = [resized_image]
        image = np.array(resized_image) / 255  # normalize pixel values between 0 and 1
        return image, resized_image[0]
    except Exception as e:
        return f"Error loading image {image_path}: {str(e)}"

def find_segment(model, image):
    return model.predict(image)

def detect_ship(predicted_masks):
    binary_mask = np.where(predicted_masks.squeeze() > threshold, 1, 0)
    return "Ship found" if np.any(binary_mask == 1) else "Ship not found"


def dice_coefficient(y_true, y_pred, smooth=1e-7):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

def load_ml_model():
    return tf.keras.models.load_model(
        'model', custom_objects={'dice_coefficient': dice_coefficient}
    )