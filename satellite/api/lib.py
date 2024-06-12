from fastapi import FastAPI
import numpy as np
from satellite.script.planet_api import PlanetDownloader
from satellite.script.preprocess import convert_npz_to_bgr
import numpy as np
from dotenv import load_dotenv
import os
from tensorflow.keras.models import load_model
from keras.metrics import MeanAbsoluteError

app = FastAPI()

# Load the .env file
load_dotenv()

# Access the environment variables
PLANET_API_KEY = os.getenv('PLANET_API_KEY')

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'ok': True}

def download_image(lat,lon):
    client = PlanetDownloader(PLANET_API_KEY)
    img = client.download_image(lat,lon, 2014, 1, 2016, 12, zoom=15)
    np.savez("image_test", img)
    return img

def prediction(image):
    custom_objects = {'mae': MeanAbsoluteError()}
    model = load_model('satellite/api/model_2.h5', custom_objects=custom_objects)
    # return model
    return model.predict(image)

@app.get('/predict')
    # Download Images for live prediction  
def predict(lat,lon):
    
    lat, lon = float(lat), float(lon)
    
    image = download_image(lat,lon)
    
    # Preprocess Image for Model
    processed_image = convert_npz_to_bgr(image)
    processed_image = processed_image/255.0
    consumption = prediction(processed_image)[0][0]
    # return {'cwd':str(os.getcwd())}
    return {'consumption' : str(consumption), 'image': str(image)}
        