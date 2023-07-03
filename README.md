# audio-sex-detection
Streamlit app for sex detection of a human voice recording
Sex detection is done through the use of a TensorFlow model after the conversion of recorded audio into a spectrogram

Audio is recorded through theevann/streamlit-audiorecorder and converted into a spectrogram using scipy and matplotlib
The spectrogram will be fed into the tensorflow model for prediction which give 2 outputs that correspond to either male or female
The higher value will show whether the voice is male or female


