import shutil
import os

path = r"D:\workspace\Hostage_Rescue_Data_FE"
folders = ["Drone", "Fountain", "Grass", "Hostage", "House", "Trailer", "Tree"]
new_path = r"D:\workspace\Hostage_Data"

for folder in folders:
    temp_path = os.path.join(new_path, folder)
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)

for folder in folders:
    temp_path = os.path.join(path, folder)
    i = 1
    for file in os.listdir(temp_path):
        if "pca" not in file and "lap" not in file:
            new_file_path = os.path.join(new_path, folder, f"{folder}_{i}.png")
            # print(f"{temp_path}\{file}     {new_file_path}")
            shutil.copyfile(os.path.join(temp_path, file), new_file_path)
            i += 1
