from fastapi import FastAPI
import matplotlib.pyplot as plt
from satellite.script.planet_api import PlanetDownloader
from satellite.script.preprocess import convert_npz_to_bgr
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import os
from tensorflow.keras.models import load_model
from keras.metrics import MeanAbsoluteError

app = FastAPI()

# Load the .env file


# Access the environment variables
PLANET_API_KEY = os.getenv('PLANET_API_KEY')

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'ok': True}

image_name = 'image.png'
image_save_path = os.path.join('satellite/api/', image_name)

def download_image(lat,lon):
    client = PlanetDownloader(PLANET_API_KEY)
    # try:
    img = client.download_image(lat,lon, 2014, 1, 2016, 12, zoom=15)
    # except:
    # print('error')
    plt.imsave(image_save_path, img)

def preprocess_image(img_path, target_size=(224, 224)):
    # Load the image from the specified file path
    img = image.load_img(img_path, target_size=target_size)

    # Convert the image to an array
    img_array = image.img_to_array(img)

    # Expand dimensions to match the expected input shape of the model (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image using VGG16's preprocess_input function
    img_array = preprocess_input(img_array)

    return img_array

def prediction(image):
    custom_objects = {'mae': MeanAbsoluteError()}
    model = load_model('satellite/api/training_model_1.h5', custom_objects=custom_objects)
    # return model
    return model.predict(image)

@app.get('/predict')
    # Download Images for live prediction  
def predict(lat,lon):
    
    lat, lon = float(lat), float(lon)
    
    download_image(lat,lon)
    
    # Preprocess Image for Model
    processed_image = preprocess_image(image_save_path)/255
    consumption = prediction(processed_image)[0][0]
    # return {'cwd':str(os.getcwd())}
    return {'consumption': str(consumption), 'image': str(processed_image)}
    
    
    # min_year = 2014
    # max_year = 2016
    # min_month = 1
    # max_month = 12
    # zoom = 15
    # image_name = 'image.png'
    # image_save_path = os.path.join('image_dir', image_name)
    # client = PlanetDownloader(PLANET_API_KEY)
    # im = client.download_image(lat, lon, min_year, min_month, max_year, max_month, zoom=zoom)
    # plt.imsave(image_save_path, im)





