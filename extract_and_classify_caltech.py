# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 07:43:37 2019

@author: USER
"""


#importig the necessary datasets
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from utilities.io import HDF5DatasetWriter
from imutils import paths
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import random
import os
import progressbar
import h5py
import pickle

random.seed(42)

#images and the rest of them
dataset_path = r"C:\Users\USER\Desktop\Data Science\My directory set-up for Computer-Vision\datasets\CALTECH-101"
output_path  = r"C:\Users\USER\Desktop\Data Science\My directory set-up for Computer-Vision\datasets\Exracted features\extracted-caltech_features"
bs = 10
buffersize = 20

#getting the list of images
imagePaths = list(paths.list_images(dataset_path))
random.shuffle(imagePaths)
labels = [p.split(os.path.sep)[-2] for p in imagePaths]
target_names = labels.copy()
le = LabelEncoder() 
labels = le.fit_transform(labels)

#importing the model and the architecture and the rest
model = VGG16(weights = "imagenet", include_top = False)
dataset = HDF5DatasetWriter((len(imagePaths), 512 * 7 * 7), output_path, dataKey = "features",
                            bufSize = buffersize )

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

#the modelling and the likes
db = h5py.File(output_path, 'a')
db["targets"] = labels
i = int(db['features'].shape[0] * 0.75)
params  = {"C":[0.1, 1, 10, 100, 1000, 10000]}
model = GridSearchCV(LogisticRegression(), params, cv = 3, verbose = 0, n_jobs = -1)
model.fit(db["features"][:i], db["targets"][:i])
print("[INFO] The best parameters are {}".format(model.best_params_))

#predicting 
preds = model.predict(db["features"][i:])
print(classification_report(db["targets"][i:],preds,target_names = list(set(target_names))))
print(confusion_matrix(db["targets"][i:], preds))
print(accuracy_score(db["targets"][i:], preds))









