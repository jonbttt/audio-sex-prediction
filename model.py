import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib
import os
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras.models import Sequential


train_ds = 'Project 2 - Audio Gender Detection/spectrograms2/train'
val_ds = 'Project 2 - Audio Gender Detection/spectrograms2/val'

train_dataset = tf.keras.utils.image_dataset_from_directory(train_ds, labels='inferred', shuffle=True, batch_size=32)
validation_dataset = tf.keras.utils.image_dataset_from_directory(val_ds, labels='inferred', shuffle=True, batch_size=32)

for example_spectrograms, example_spect_labels in train_dataset.take(1):
  break

class_names = ['female', 'male']

norm_layer = tf.keras.layers.experimental.preprocessing.Normalization()
norm_layer.adapt(data=train_dataset.map(map_func=lambda spec, label: spec))

input_shape = example_spectrograms.shape[1:]
model = models.Sequential([
    layers.Input(shape=input_shape),
    tf.keras.layers.experimental.preprocessing.Rescaling(32, 32),
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2),
])

model.summary()

model.compile(
    optimizer='Adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
  filepath=checkpoint_path,
  save_weights_only=True,
  verbose=1
)

EPOCHS = 1
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

model.save('C:/Users/User/Jonathan/Project 2 - Audio Gender Detection/models/test.h5')