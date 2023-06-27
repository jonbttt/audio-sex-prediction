import splitfolders
data_dir = 'Project 2 - Audio Gender & Emotion Detection/spectrograms'
splitfolders.ratio(data_dir, output="Project 2 - Audio Gender & Emotion Detection/spectrograms2", seed=1337, ratio=(.8, 0.1,0.1))