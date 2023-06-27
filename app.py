import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder
from converter import convert
from converter import predict

st.title('Audio Gender Detection')
audio_bytes = audio_recorder()
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
if st.button('Press button to re-record'):
    st.experimental_rerun
else:
    pass

if st.button('Press button to continue'):
    spectrogram = convert(audio_bytes)
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