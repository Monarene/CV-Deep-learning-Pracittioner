# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:08:33 2019

@author: Michael
"""

#importig the necessary datasets
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from utilities.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import random
import os
import progressbar
random.seed(42)