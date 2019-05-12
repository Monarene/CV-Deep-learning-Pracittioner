# -*- coding: utf-8 -*-
"""
Created on Fri May 10 23:07:43 2019

@author: H P ENVY
"""

#importing the neccessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utilities.preprocessing import ImageToArrayPreprocessor, AspectAwarePreprocessor
from utilities.datasets import SimpleDatasetLoader
from utilities.nn.cnn import FCHeadNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop, SGD
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
from imutils import paths
import numpy as np
import os
import cv2

#prepare the image data generator for data augmentaion
aug = ImageDataGenerator(rotation_range=30, width_shift_range = 0.1, height_shift_range = 0.1,
                         shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, fill_mode = "nearest")

#preparing the data and the list of images
imagePaths = list(paths.list_images(r"C:\Users\H P ENVY\Desktop\Data Science\My directory set-up for Computer-Vision\datasets\Flowers17"))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]
aap = AspectAwarePreprocessor(224,224)
isp = ImageToArrayPreprocessor()

#load the dataset from disk and then 
sdl = SimpleDatasetLoader(preprocessors = [aap, isp])
(data, labels) = sdl.load(imagePaths, verbose = 1)
data = data.astype("float") / 255.0

















