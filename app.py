from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import pickle
import cv2
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect():
    if request.method == 'POST':
        # Load model
        MODEL_PATH = "myvgg16_model.h5"
        # Load pickle (ini adalah nama class atau label yang digunakan)
        PICKLE_PATH = "class_names.pkl"

        # Isi dari class_names.pkl: 
        # class_names = ['Aster', 'Euphorbia', 'Bergamot', 'Sage', 'Azalea', 'Peony', 'Pansy', 'Orchid', 'Dandelion', 'Cosmos', 'Snapdragon', 'Polyanthus', 'Dahlia', 'Gerbera', 'Ixora', 'Eustoma', 'Daisy', 'Aglaonema', 'Sunflower', 'Viola', 'Lily Flower', 'Rose', 'Iris', 'Tuberose', 'Alyssum', 'Dieffenbacia', 'Jasmine', 'Lavender', 'Tulip']

        filestr = request.files['image'].read()
        npimg = np.fromstring(filestr, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # pre-process the image for classification
        image = cv2.resize(image, (224, 224))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # load the trained CNN and the label binarizer
        print("[INFO] loading network...")
        model = load_model(MODEL_PATH)
        lb = pickle.loads(open(PICKLE_PATH, "rb").read())

        # classify the input image
        print("[INFO] classifying image...")
        proba = model.predict(image)[0]
        idx = np.argmax(proba)
        label = lb[idx]

        return jsonify(success=1, label=label, percent=(proba[idx] * 100))
    
    return jsonify(error=1, message='Unsupported HTTP method')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
