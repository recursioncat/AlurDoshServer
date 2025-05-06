from flask import Flask, request, jsonify
import keras
from PIL import Image
import numpy as np
from dependencies import processImageFromPIL, predictPotato, predictTomato

app = Flask(__name__)

@app.route('/classifyUnknown', methods=['POST'])
def classify():
    model = keras.models.load_model('./Models/Classifier.keras')
    possiblePredictions = ['Potato', 'Tomato']

    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image = Image.open(file.stream).convert('RGB')
    image_array = processImageFromPIL(image, 128)
    prediction = possiblePredictions[np.argmax(model.predict(image_array))]

    veggie_func = globals().get('predict' + prediction)
    if not veggie_func:
        return jsonify({'error': f"No prediction function for {prediction}"}), 500

    disease = veggie_func(image)
    return jsonify({'vegType': prediction, 'disease': disease})


app.run(debug=True)
