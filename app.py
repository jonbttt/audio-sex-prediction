import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from audiorecorder import audiorecorder
from converter import convert
from converter import predict

st.title('Audio Gender Detection')
st.title("Audio Recorder")
audio = audiorecorder("Click to record", "Recording...")
if len(audio) > 0:
    wav_file = open("audio.mp3", "wb")
    wav_file.write(audio.tobytes())
if st.button('Press button to continue'):
    spectrogram = convert(wav_file)
    values = predict(spectrogram)
    values = str(values)
    values = values.replace('[[ ','')
    values = values.replace(']]','')
    values = list(values.split(' '))
    if values[0] > values[1]:
        gender = 'Female'
    elif values[0] < values[1]:
        gender = 'Male'
    elif values[0] == values[1]:
        gender = 'Unknown'
    st.subheader("M/F: "+gender)
else:
    pass
