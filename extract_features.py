# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 17:31:50 2019

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

#the neccesary variables
datasetPath = r"C:\Users\USER\Desktop\Data Science\My directory set-up for Computer-Vision\datasets\animals"
outPutPath = r"C:\Users\USER\Desktop\Data Science\My directory set-up for Computer-Vision\Deep-Learning-for-Computer-Vision\Deep learning for computer vision - Practitioneers bundle\Exracted features\extract_animals_vgg16"
bs = 10
bufferSize = 20

#dealing the imagePaths and all the underlying properties
imagePaths = list(paths.list_images(datasetPath))
random.shuffle(imagePaths)
labels = [p.split(os.path.sep)[-2] for p in imagePaths]

le = LabelEncoder()
labels = le.fit_transform(labels)

#importing the model
model = VGG16(weights = "imagenet", include_top = False)
dataset = HDF5DatasetWriter((len(imagePaths), 512 * 7 * 7), outPutPath, dataKey = "features",
                            bufSize = bufferSize)
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
    features = features.reshape((features.shape[0], 512 * 7 * 7))
    dataset.add(features, batchLabels)

dataset.close()





