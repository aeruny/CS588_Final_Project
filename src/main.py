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
from sklearn.decomposition import PCA


img = imageio.imread('s1/1.pgm')
img = img.astype(np.uint8)
img = img / 255
plt.imshow(img,cmap='gray')