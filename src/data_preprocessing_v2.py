import os
import time

import numpy as np
from PIL import Image

t_total = time.time()

data_root = r"D:\workspace\Hostage_Data"
new_path = r"D:\workspace\Hostage_Data_Preprocessed"
folders = ["Drone", "Hostage", "House", "Tree"]
img_size = 64


for folder in folders:
    temp_path = os.path.join(new_path, folder)
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)


for folder in folders:
    print(f"Looking at the {folder} folder")
    t_folder = time.time()
    image_data = []
    target_labels = []
    for file in os.listdir(os.path.join(data_root, folder)):
        # Open image file
        file_path = os.path.join(data_root, folder, file)
        image = Image.open(file_path)

        # Rescale
        image = image.resize((img_size, img_size))

        # Save the preprocessed images
        image.save(os.path.join(new_path, folder, file))

        # Flatten Numpy Array
        image_array = np.asarray(image).flat
