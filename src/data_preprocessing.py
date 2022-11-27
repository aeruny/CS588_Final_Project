import os
import time

import numpy as np
import pandas as pd
from PIL import Image

t_total = time.time()

data_root = r"D:\workspace\Hostage_Data"
new_path = r"D:\workspace\Hostage_Data_Preprocessed"
# folders = ["Drone", "Fountain", "Grass", "Hostage", "House", "Trailer", "Tree"]
folders = ["Drone", "Hostage", "House", "Tree"]
enable_rescale = True
rescale_factor = 2
img_size = 512

img_size = img_size // rescale_factor if enable_rescale else img_size

for folder in folders:
    temp_path = os.path.join(new_path, folder)
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)


image_data_label = [f"{a}x{b}" for a in range(1, img_size + 1) for b in range(1, img_size + 1)]
for folder in folders:
    print(f"Looking at the {folder} folder")
    t_folder = time.time()
    image_data = []
    target_labels = []
    for file in os.listdir(os.path.join(data_root, folder)):
        # Open image file
        file_path = os.path.join(data_root, folder, file)
        image = Image.open(file_path)

        # Grayscale
        image = image.convert('L')

        # Rescale
        if enable_rescale:
            image = image.resize((img_size, img_size))

        # Save the preprocessed images
        image.save(os.path.join(new_path, folder, file))

        # Flatten Numpy Array
        image_array = np.asarray(image).flat

        # Append image data and label
        image_data.append(image_array)
        target_labels.append(folder)
    print(f"{time.time() - t_folder} seconds have passed in the {folder} folder")
    print(f"Creating a dataframe for {folder} data")
    t_df = time.time()

    df = pd.DataFrame(data=image_data, columns=image_data_label)
    df.insert(loc=df.shape[-1], column="Target", value=target_labels)
    df.to_pickle(f"./{folder}_dataset.pkl")
    print(f"{time.time() - t_df} seconds have passed while creating the {folder} dataset")


# print("Concatenating all dataframes")
# drone_df = pd.read_pickle("./Drone_dataset.pkl")
# hostage_df = pd.read_pickle("./Hostage_dataset.pkl")
# house_df = pd.read_pickle("./House_dataset.pkl")
# tree_df = pd.read_pickle("./Tree_dataset.pkl")
#
# frames = [drone_df, hostage_df, house_df, tree_df]
# dataset_df = pd.concat(frames)
# print(dataset_df.shape)
# print(dataset_df)
# dataset_df.to_pickle("dataset.pkl")

print(f"{time.time() - t_total} seconds have passed in total")
