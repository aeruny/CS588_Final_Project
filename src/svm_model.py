import os

import pickle
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data_root = r"D:\workspace\Hostage_Data_Preprocessed"
categories = {"Drone": 0, "Hostage": 1, "House": 2, "Tree": 3}

training_data = []
for folder in categories:
    print(f"Checking images from {folder} folder")
    for image in os.listdir(os.path.join(data_root, folder)):
        # Open image file
        image_path = os.path.join(data_root, folder, image)
        img_array = cv2.imread(image_path)
        training_data.append([img_array, categories[folder]])

print(f"# of images: {len(training_data)}")

X = []
y = []

# Compose the dataset
for categories, label in training_data:
    X.append(categories)
    y.append(label)
X = np.array(X).reshape(len(training_data), -1)
y = np.array(y)

# Flatten the dataset
X = X / 255.0

print(f"X.shape = {X.shape},  y.shape = {y.shape}")

# Dataset Split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

print(f"X_train: {X_train.shape},  y_train: {y_train.shape}")
print(f"X_test: {X_test.shape},  y_test: {y_test.shape}")

# Training SVM Model
print("Training SVC Model")
svc = SVC(kernel='linear', gamma='auto')
svc.fit(X_train, y_train)

print("Testing SVM Model")
y2 = svc.predict(X_test)

print(f"Accuracy of SVM Model is {accuracy_score(y_test, y2)}")
print(classification_report(y_test, y2))

# Save SVM Model
print("Saving the SVM Model")
with open('svc_resize_64.pkl', 'wb') as f:
    pickle.dump(svc, f)


result = pd.DataFrame({'original': y_test[:100], 'predicted': y2[:100]})

print(result)


