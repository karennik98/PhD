import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import Utils

def load():
    return load_model(Utils.model_path)

def predict(model, data):
    return model.predict(data)
