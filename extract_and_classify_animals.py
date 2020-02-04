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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import random
import os
import progressbar
import h5py
import pickle

random.seed(42)

#the neccesary variables
dataset_path = r"C:\Users\USER\Desktop\Data Science\My directory set-up for Computer-Vision\datasets\animals"
output_path = r"C:\Users\USER\Desktop\Data Science\My directory set-up for Computer-Vision\Deep-Learning-for-Computer-Vision\Deep learning for computer vision - Practitioneers bundle\Exracted features\extract_animals_vgg16"
model_path = r"C:\Users\H P ENVY\Desktop\Data Science\Soil Analysis Propject\models\first_model.pickle"

bs = 10
bufferSize = 20

#dealing the imagePaths and all the underlying properties
imagePaths = list(paths.list_images(dataset_path))
random.shuffle(imagePaths)
labels = [p.split(os.path.sep)[-2] for p in imagePaths]

le = LabelEncoder()
labels = le.fit_transform(labels)

#importing the model
model = VGG16(weights = "imagenet", include_top = False)
dataset = HDF5DatasetWriter((len(imagePaths), 512 * 7 * 7), output_path, dataKey = "features",
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

#modelling
db = h5py.File(output_path, "a")
db["targets"] = labels
i = int(db["labels"].shape[0] * 0.6)
#params = {'C':np.arange(0.01,10,0.1)}
params = {"learning_rate":np.arange(0.05,1,0.05), "n_estimators":np.arange(10,200,10),
          "max_features":[1,2,3], "max_depth":[1,2,3], "random_state":np.arange(1,100)}
#model_2 = RandomizedSearchCV(GradientBoostingClassifier(), params, n_iter = 50, cv = 5, verbose = 0, n_jobs = -1)
#model = GridSearchCV(LogisticRegression(), params, cv = 3, verbose = 0, n_jobs = -1)
model = GridSearchCV(GradientBoostingClassifier(), params, cv = 3, verbose = 0, n_jobs = -1)
model.fit(db["features"][:i], db["targets"][:i])
print("[INFO] Best parameters:{}".format(model.best_params_))

#checking and other things
print("[INFO] Evaluating predictions...")
preds = model.predict(db["features"][i:])
print(classification_report(db["targets"][i:], preds, target_names = ["Cat", "Dog", "Panda"])).

#saving the model
f = open(model_path, "wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()

422





















