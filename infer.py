import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

# Function to resize an image
def resize_image(image, size=(224, 224)):
    resized_img = cv2.resize(image, size)
    return resized_img


# Function to preprocess an input image
def preprocess_input_image(image):
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.mobilenet.preprocess_input(image_array)
    return image_array

# Function to perform image segmentation prediction
def predict_image_segmentation(model, image):
    input_image = preprocess_input_image(image)
    prediction = model.predict(input_image)
    binary_mask = prediction > 0.5
    return binary_mask[0, :, :, 0]

# Load the pre-trained model
model_config_path = r'tech\final_best1.h5'  # Replace with the actual path to your model
model = tf.keras.models.load_model(model_config_path)

# Load an example image
image_path = 'IMG_5526.jpg'  # Replace with the actual path to your image
image = cv2.imread(image_path)

# Perform inference on the image
resized_image = resize_image(image)
predicted_mask = predict_image_segmentation(model, resized_image)

# Check if any True value exists in the predicted mask
if np.any(predicted_mask):
    print("The object is present")
else:
    print("The object is not present")
