from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import json
import requests
from keras.preprocessing.image import img_to_array
from PIL import Image
import io

app = Flask(__name__)

# Ganti dengan API key dan URL deployment Anda
API_KEY = "<apikey>"
SCORING_URL = "<endpoint>"

# Class names sesuai dengan model Anda
class_names = ['Dahlia', 'Dieffenbacia', 'Daisy', 'Lily Flower', 'Orchid', 'Azalea', 'Iris', 'Ixora', 'Sage', 'Tuberose', 'Jasmine', 'Bergamot', 'Aster', 'Gerbera', 'Lavender', 'Eustoma', 'Dandelion', 'Cosmos', 'Euphorbia', 'Viola', 'Peony', 'Snapdragon', 'Tulip', 'Alyssum', 'Rose', 'Polyanthus', 'Pansy', 'Sunflower', 'Aglaonema']
class_names.sort()

def get_token():
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }
    data = f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={API_KEY}"
    response = requests.post(url, headers=headers, data=data)
    response.raise_for_status()
    return response.json()["access_token"]

def api_post(scoring_url, token, payload):
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json;charset=UTF-8"
    }
    response = requests.post(scoring_url, headers=headers, data=payload)
    response.raise_for_status()
    return response.json()

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image.tolist()

@app.route('/detect', methods=['POST'])
def detect():
    if request.method == 'POST':
        filestr = request.files['image'].read()
        image = Image.open(io.BytesIO(filestr))

        # Preprocess the image for classification
        processed_image = preprocess_image(image, target_size=(224, 224))

        # Get the token
        token = get_token()

        # Create payload
        payload = json.dumps({
            "input_data": [{
                "fields": ["image"],
                "values": processed_image
            }]
        })

        # Call the IBM Cloud API
        prediction = api_post(SCORING_URL, token, payload)
        
        # Extract the prediction results
        proba = prediction["predictions"][0]["values"][0]
        idx = np.argmax(proba)
        label = class_names[idx]

        return jsonify(success=1, label=label, percent=(proba[idx] * 100))
    
    return jsonify(error=1, message='Unsupported HTTP method')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)