import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the model
model_path = 'MyModel.keras'  # Adjust the path to your model file
model = tf.keras.models.load_model(model_path)
model.summary()  # Print model summary for debugging

print('Model loaded. Check http://127.0.0.1:5000/')

# Define labels in the same order as the model output
labels = {
    0: 'Apple',
    1: 'Banana',
    2: 'beetroot',
    3: 'bell pepper',
    4: 'cabbage',
    5: 'capsicum',
    6: 'carrot',
    7: 'cauliflower',
    8: 'chilli pepper',
    9: 'corn',
    10: 'cucumber',
    11: 'eggplant',
    12: 'garlic',
    13: 'ginger',
    14: 'grapes',
    15: 'jalepeno',
    16: 'kiwi',
    17: 'lemon',
    18: 'lettuce',
    19: 'mango',
    20: 'onion',
    21: 'orange',
    22: 'paprika',
    23: 'pear',
    24: 'peas',
    25: 'pineapple',
    26: 'pomegranate',
    27: 'potato',
    28: 'raddish',
    29: 'soy beans',
    30: 'spinach',
    31: 'sweetcorn',
    32: 'sweetpotato',
    33: 'tomato',
    34: 'turnip',
    35: 'watermelon'
}

# Function to preprocess the image and get predictions
def getResult(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Assuming input shape is (224, 224)
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)

    predictions = model.predict(x)
    return predictions[0]

# Routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predictApi', methods=["POST"])
def api():
    try:
        if 'file' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('file')
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(image.filename))
        image.save(file_path)
        predictions = getResult(file_path)
        predicted_label = labels[np.argmax(predictions)]
        return jsonify({'prediction': predicted_label})
    except Exception as e:
        return jsonify({'Error': str(e)})

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        predictions = getResult(file_path)
        predicted_label = labels[np.argmax(predictions)]
        return predicted_label

    return None

if __name__ == '__main__':
    app.run(debug=True)
