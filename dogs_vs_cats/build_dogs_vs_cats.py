# -*- coding: utf-8 -*-
"""
Created on Sat May 18 10:44:21 2019

@author: H P ENVY
"""

#importing the necessary packages
from config import dogs_vs_cats_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utilities.preprocessing import AspectAwarePreprocessor
from utilities.io import hdf5_dataset_writer
from imutils import paths
import os
import numpy as np
import json
import cv2

#time to grab the path to the images and all
trainPaths = list(paths.list_images(config.IMAGES_PATH))
trainLabels  = [p.split(os.path.sep)[-1].split(".")[0] for p in config.IMAGES_PATH]

