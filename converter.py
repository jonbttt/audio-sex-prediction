import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import tensorflow as tf
import io
from scipy import signal
from scipy.io import wavfile
from io import BytesIO
from PIL import Image

def convert(file):
    sample_rate, samples = wavfile.read(file)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    plt.pcolormesh(times, frequencies, spectrogram)
    fig1 = plt.imshow(spectrogram)
    plt.axis('off')
    fig1.axes.get_xaxis().set_visible(False)
    fig1.axes.get_yaxis().set_visible(False)
    plt.grid(False)
    fig = plt.gcf()
    fig.set_size_inches(8,1.5)
    fig.set_dpi(100)
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    img = Image.open(io_buf)
    img256 = img.resize((256,256), Image.ANTIALIAS)
    imgrgb = img256.convert('RGB')
    np_image_data = np.asarray(imgrgb)
    print(np_image_data.shape)
    return np_image_data

def predict(ndarray):
    model_path = 'C:/Users/User/Jonathan/Project 2 - Audio Gender Detection/models/audiomodel2.h5'
    model = tf.keras.models.load_model(model_path)
    results = model.predict(ndarray.reshape(-1, 256, 256, 3))

    return results