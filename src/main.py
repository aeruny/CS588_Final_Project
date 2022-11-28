import imageio.v2 as imageio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os

import scipy
import sklearn
import cv2
from PIL import Image
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Flatten
from keras.regularizers import l2

train_dir = "dataset/training_set/"
test_dir = "dataset/test_set/"

folders = ["Drone", "Hostage", "House", "Tree"]

train_datagen = ImageDataGenerator(rescale=(1 / 255.), shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
training_set = train_datagen.flow_from_directory(directory=train_dir, target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode="binary")
test_datagen = ImageDataGenerator(rescale=(1 / 255.))
test_set = test_datagen.flow_from_directory(directory=test_dir, target_size=(64, 64),
                                            batch_size=32,
                                            class_mode="binary")

model = Sequential()
model.add(Conv2D(filters=32, padding="same", activation="relu", kernel_size=3, strides=2, input_shape=(128, 128, 3)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=32, padding="same", activation="relu", kernel_size=3))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Flatten())
model.add(Dense(128, activation="relu"))

# Output layer
model.add(Dense(1, kernel_regularizer=l2(0.01), activation="linear"))

model.add(Dense(len(folders), kernel_regularizers=l2(0.01), activation="softmax"))
model.compile(optimizer="adam", loss="squared_hinge", metrics=['accuracy'])

history = model.fit(x = training_set, validation_data = test_set, epochs=15)