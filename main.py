from flask import Flask, request, render_template, redirect, url_for, session
import os
import pandas as pd
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import openai
from functools import wraps

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['IMAGES_FOLDER'] = 'static/images'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# OpenAI API key for image generation
openai.api_key = ''

app.secret_key = os.urandom(24)

# Load user credentials
def load_credentials():
    try:
        df = pd.read_csv('database.csv')
        return df.set_index('username')['password'].to_dict()
    except FileNotFoundError:
        return {}

# Load the feature list and filenames for fashion recommendation
try:
    feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
except FileNotFoundError:
    feature_list = np.array([])

image_filenames = [f for f in os.listdir(app.config['IMAGES_FOLDER']) if
                   os.path.isfile(os.path.join(app.config['IMAGES_FOLDER'], f))]

print(f"Loaded {len(feature_list)} features.")
print(f"Found {len(image_filenames)} images.")

# Define the model for feature extraction
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    if len(feature_list) == 0:
        return np.array([[]]), np.array([[]])
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices, distances

def generate_image(prompt):
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="512x512"
    )
    image_url = response['data'][0]['url']
    description = f"This is an image related to: {prompt}"
    return image_url, description

credentials = load_credentials()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in credentials and credentials[username] == password:
            session['username'] = username
            return redirect(url_for('homepage'))
        else:
            return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')

@app.route('/')
@app.route('/index')
@login_required
def index():
    return render_template('index.html')


@app.route('/homepage')
@login_required
def homepage():
    return render_template('homepage.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend_route():
    recommendations = []
    uploaded_image = None
    chatbot_response = None
    generated_image_url = None
    generated_description = None

    if request.method == 'POST':
        # Handle image upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '' and allowed_file(file.filename):
                filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filename)
                # Feature extraction and recommendation
                features = feature_extraction(filename, model)
                indices, distances = recommend(features, feature_list)
                recommendations = [image_filenames[i] for i in indices[0] if i < len(image_filenames)]
                uploaded_image = file.filename

        # Handle chatbot query
        elif 'chat_input' in request.form:
            user_input = request.form['chat_input']
            if user_input:
                generated_image_url, generated_description = generate_image(user_input)
                chatbot_response = {
                    'image_url': generated_image_url,
                    'description': generated_description
                }

    return render_template('index.html', uploaded_image=uploaded_image, recommendations=recommendations,
                           chatbot_response=chatbot_response)

@app.route('/logout')
@login_required
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, port=8000)
