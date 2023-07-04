# audio-sex-prediction
Streamlit app for sex prediction of a human voice recording

Sex prediction is done through the use of a TensorFlow model trained on spectrograms from sample audio recordings

##Process##
Audio is recorded through theevann/streamlit-audiorecorder and converted into a spectrogram using Scipy and Matplotlib.
The spectrogram will be fed into the TensorFlow model for prediction which gives 2 outputs that correspond to either male or female.
The higher value will show whether the voice is male or female.

##Result:##
![alt text](https://github.com/jonbttt/audio-sex-prediction/blob/main/test-result.png?raw=true)
