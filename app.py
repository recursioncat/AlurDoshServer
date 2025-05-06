from flask import Flask, request, jsonify, redirect
from tensorflow import keras
from dependencies import *
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)

@app.route('/classifyUnknown', methods=['POST'])
def classify():

    model = keras.models.load_model('Models/Classifier.keras')
    possiblePredictions = ['Potato', 'Tomato']
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image = Image.open(file.stream).convert('RGB')
    image = processImage(image, 128)  # modify processImage to accept PIL images
    prediction = possiblePredictions[np.argmax(model.predict(image))]

    veggie = globals().get('predict'+prediction)
    result = {'vegType': prediction, 'disease': veggie(file)}
    return jsonify(result)



@app.route('/potato')
def potato():
    data = request.get_json(force = True)
    url = data['url']
    return predictPotato(url)
    
@app.route('/tomato')
def tomato():
    data = request.get_json(force = True)
    url = data['url']
    return predictTomato(url)

@app.route('/')
def hello():
    return "Hello from the Classification Server"
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
 