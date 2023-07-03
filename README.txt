Streamlit app for sex detection of a voice recording
Sex detection is done through the use of a TensorFlow model after the conversion of recorded audio into a spectrogram

Audio is recorded through theevann/streamlit-audiorecorder and converted into a spectrogram using scipy and matplotlib
The spectrogram will then be fed into the tensorflow model which will give 2 outputs that correspond to either male or female.
The higher value will show whether the voice is male or female


