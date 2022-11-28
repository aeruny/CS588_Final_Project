import os

import cv2
import numpy as np


def get_dataset():
    data_root = r"D:\workspace\Hostage_Data_Preprocessed"
    categories = {"Drone": 0, "Hostage": 1, "House": 2, "Tree": 3}
    training_data = []
    for folder in categories:
        print(f"Retrieving images from {folder} folder")
        for image in os.listdir(os.path.join(data_root, folder)):
            # Open image file
            image_path = os.path.join(data_root, folder, image)
            img_array = cv2.imread(image_path)
            training_data.append([img_array, categories[folder]])
    X = []
    y = []

    # Compose the dataset
    for categories, label in training_data:
        X.append(categories)
        y.append(label)
    X = np.array(X).reshape(len(training_data), -1)
    y = np.array(y)

    return X, y