#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 15:18:12 2019

@author: michael
"""

#importing the necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from keras.applications import ResNet50, imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
from utilities.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import random
import os
import pickle

#setting the argument variables
datasetPath = r"/home/michael/Desktop/Datascience Projects/My directory set-up for Computer-Vision/datasets/kaggle_dogs_vs_cats/train/Kaggle_original_data"
outputDataset = r"/home/michael/Desktop/Datascience Projects/My directory set-up for Computer-Vision/datasets/kaggle_dogs_vs_cats/hdf5/fine_tuning_resnet.hdf5"
bs = 10
buffer_size = 20


#data dealing
imagePaths = list(paths.list_images(datasetPath))
random.shuffle(imagePaths)
labels = [p.split(os.path.sep)[-1].split(".")[0] for p in imagePaths]
le  = LabelEncoder()
labels = le.fit_transform(labels)
model = ResNet50(weights = "imagenet", include_top = False)
dataset = HDF5DatasetWriter((len(imagePaths), 100352), outputDataset, dataKey = "features",
                            bufSize = buffer_size)
dataset.storeClassLabels(le.classes_)

#looping over all the images, preprocessing them, and storing them in hdf5
for i in np.arange(0, len(imagePaths), bs):
    batchPaths = imagePaths[i: i + bs]
    batchLabels = labels[i: i + bs]
    batchImages = []
    
    for imagePath in batchPaths:
        image = load_img(imagePath, target_size = (224,224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis = 0)
        image = imagenet_utils.preprocess_input(image)
        batchImages.append(image)

    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size = bs)
    features = features.reshape((features.shape[0], 100352))
    dataset.add(features, batchLabels)

dataset.close()
























