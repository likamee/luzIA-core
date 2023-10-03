from io import BytesIO

import numpy as np
import requests
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

from app.utilities import cam_models


def preprocess_image(image_url, target_size):
    """Preprocess the input image for evaluation."""
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image = image.resize(target_size, Image.ANTIALIAS)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image


def evaluate_image(model, image_url, target_size=(224, 224)):
    """Evaluate an image using the provided model."""
    image = preprocess_image(image_url, target_size)
    prediction = model.predict(image)
    return prediction


def process_evaluate(cfg, **kwargs):
    pato = kwargs['pato']
    filter = kwargs['filter']

    model_path = 'models/{}/{}_{}_{}.hdf5'.format(cfg.ds, pato, filter, cfg.n_layers)
    image_url = 'https://example.com/path/to/image.jpg'

    model = cam_models.load_saved_model(model_path)
    prediction = evaluate_image(model, image_url)

    print("Prediction:", prediction)
