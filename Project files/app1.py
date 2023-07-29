import numpy as np
import os
import requests
from flask import Flask, request, render_template

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Load the model
model = load_model(r"E:\IBM\ref\VGG16\Flask\vgg-16-Tea-leaves-disease-model.h5")

# Default home page or route
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/teahome')
def teahome():
    return render_template('teahome.html')

@app.route('/teapred', methods=["GET", "POST"])
def teapred():
    return render_template('teapred.html')

@app.route('/tearesult', methods=["POST"])
def tearesult():
    if request.method == "POST":
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        f.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)
        prediction = np.argmax(model.predict(img_data))

        index = ['Anthracnose', 'algal leaf', 'bird eye spot', 'brown blight', 'gray Light', 'healthy',
                 'red Leaf spot', 'white spot']
        nresult = index[prediction]

        return render_template('teapred.html', prediction=nresult)

if __name__ == '__main__':
    app.run(debug=False, port=8080)