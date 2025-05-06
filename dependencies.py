from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import keras
import pickle

# DEPRECATED: Only for file paths
def processImage(path, size):
    img = keras_image.load_img(path, target_size=(size, size))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ✅ For PIL images
def processImageFromPIL(image, target_size):
    image = image.convert("RGB")
    image = image.resize((target_size, target_size))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ✅ Updated to accept PIL image
def predictPotato(pil_image):
    model = keras.models.load_model('Models/Potatoes.keras')
    image = processImageFromPIL(pil_image, 128)
    prediction = model.predict(image)
    with open('Labels/potato.label', 'rb') as file:
        labels = pickle.load(file)
    return str(labels[np.argmax(prediction)])

# ✅ Updated to accept PIL image
def predictTomato(pil_image):
    model = keras.models.load_model('Models/Tomato.keras')
    image = processImageFromPIL(pil_image, 128)
    prediction = model.predict(image)
    with open('Labels/tomato.label', 'rb') as file:
        labels = pickle.load(file)
    return str(labels[np.argmax(prediction)])
