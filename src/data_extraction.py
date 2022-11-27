import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import sklearn
import cv2
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

data_root = r"D:\workspace\Hostage_Data_Preprocessed"
folders = ["Drone", "Hostage", "House", "Tree"]
thing = { "Drone": 0, "Hostage": 1, "House": 2, "Tree": 3 }

images = []
labels = []
for folder in folders:
    temp_path = os.path.join(data_root, folder)
    for file in os.listdir(temp_path):
        image = Image.open(os.path.join(temp_path, file))
        images.append(np.asarray(image))
        labels.append(folder)

labels = [thing[label] for label in labels]

images = np.asarray(images)
labels = np.asarray(labels)

train_images, test_images = train_test_split(images, train_size=0.7)
train_labels, test_labels = train_test_split(labels, train_size=0.7)

train_images = (np.expand_dims(train_images, axis=-1) / 255.).astype(np.float32)
train_labels = (train_labels / 3).astype(np.float32)
test_images = (np.expand_dims(test_images, axis=-1) / 255.).astype(np.float32)
test_labels = (test_labels / 3).astype(np.float32)

# print(train_images)
print(train_labels)
# print(test_images)
print(test_labels)


