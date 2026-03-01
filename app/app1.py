
from googletrans import Translator
from flask import Flask, render_template, request, redirect, url_for, flash
from markupsafe import Markup
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
# import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9

# Authentication imports
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt

translator = Translator()


disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

import os

# Get absolute path to the directory this file is in
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_model(model_path_relative):
    """
    Tries to find a model file in several possible locations.
    :param model_path_relative: path like 'models/plant_disease_model.pth'
    :return: Absolute path to the model if found, else original relative path
    """
    # 1. Check relative to current file (app/models/...)
    path1 = os.path.join(BASE_DIR, model_path_relative)
    if os.path.exists(path1):
        return path1
    
    # 2. Check relative to root (models/...)
    path2 = os.path.join(os.path.dirname(BASE_DIR), model_path_relative)
    if os.path.exists(path2):
        return path2
    
    # 3. Check absolute paths for Hugging Face container
    path3 = os.path.join('/app', model_path_relative)
    if os.path.exists(path3):
        return path3
    
    path4 = os.path.join('/app/app', model_path_relative)
    if os.path.exists(path4):
        return path4
        
    # Default fallback
    return path1

disease_model_path = find_model('models/plant_disease_model.pth')
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


from joblib import load

crop_recommendation_model = load(find_model('models/RF.joblib'))


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app1 = Flask(__name__)
app1.config['SECRET_KEY'] = config.SECRET_KEY
app1.config['SQLALCHEMY_DATABASE_URI'] = config.SQLALCHEMY_DATABASE_URI

db = SQLAlchemy(app1)
bcrypt = Bcrypt(app1)
login_manager = LoginManager(app1)
login_manager.login_view = 'login'

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

with app1.app_context():
    db.create_all()

# Auth Routes
@app1.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        try:
            name = request.form.get('name')
            email = request.form.get('email')
            password = request.form.get('password')
            
            # Check if user already exists
            user_exists = User.query.filter_by(email=email).first()
            if user_exists:
                flash('Email address already exists', 'danger')
                return redirect(url_for('signup'))
                
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            user = User(name=name, email=email, password=hashed_password)
            db.session.add(user)
            db.session.commit()
            flash('Your account has been created! You are now able to log in', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            print(f"Error during signup: {e}")
            flash('An error occurred during signup. Please try again.', 'danger')
            return redirect(url_for('signup'))
    return render_template('signup.html', title='Sign Up')

@app1.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user, remember=True)
            print(f"User {email} logged in successfully. Authenticated: {current_user.is_authenticated}")
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login')

@app1.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('home'))

# render home page


@ app1.route('/')
def home():
    print(f"Home page access. User authenticated: {current_user.is_authenticated}")
    title = 'Harvestify - Home'
    return render_template('index.html', title=title)

# render crop recommendation form page


@ app1.route('/crop-recommend')
@login_required
def crop_recommend():
    title = 'Harvestify - Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page


@ app1.route('/fertilizer')
@login_required
def fertilizer_recommendation():
    title = 'Harvestify - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)

@ app1.route('/crop-predict', methods=['POST'])
@login_required
def crop_prediction():
    title = 'Harvestify - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")
        language = request.form.get("language")
        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]
            crop_recommendation = translate_text(
                f"You should grow {final_prediction} in your farm.", language)
            return render_template('crop-result.html', prediction=crop_recommendation, title=title)
        else:
            return render_template('try_again.html', title=title)


# render fertilizer recommendation result page

from markupsafe import Markup
import pandas as pd

# Import fertilizer dictionary
from utils.fertilizer import fertilizer_dic
@ app1.route('/fertilizer-predict', methods=['POST'])
@login_required
def fert_recommend():
    title = 'Harvestify - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    language = request.form.get("language")
    # Read fertilizer data
    df = pd.read_csv('FertilizerData.csv')

    # Get nutrient levels for the specified crop
    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    # Calculate differences in nutrient levels
    n = nr - N
    p = pr - P
    k = kr - K

    # Determine the key based on the maximum difference
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    # Get recommendation from the fertilizer dictionary based on the key
    recommendation = Markup(str(fertilizer_dic[key]))
    translated_recommendation = translate_text(recommendation, language)

    # Pass translated recommendation to template
    return render_template('fertilizer-result.html', recommendation=translated_recommendation, title=title)


def translate_text(text, target_language):
    # Translate text to the target language using Google Translate API
    translated_text = translator.translate(text, dest=target_language)
    return translated_text.text



from utils.disease import disease_dic

@app1.route('/disease-predict', methods=['GET', 'POST'])
@login_required
def disease_prediction():
    title = 'Harvestify - Disease Detection'
    language = request.form.get("language")
    translated_recommendation = ""  # Define the variable outside of the try block
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()
            prediction = predict_image(img)
            prediction = Markup(str(disease_dic[prediction]))
            translated_recommendation = translate_text(prediction, language)
        except:
            pass
        return render_template('disease-result.html', prediction=translated_recommendation, title=title)
    return render_template('disease.html', title=title)



# ===============================================================================================
if __name__ == '__main__':
    app1.run(debug=False)