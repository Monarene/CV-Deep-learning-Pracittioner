# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:34:35 2019

@author: H P ENVY
"""

#importing the necessasry libnrariea
from config import dogs_vs_cats_config as config
from utilities.preprocessing import CropPreprocessor
from utilities.preprocessing import MeanPreprocessor, ImageToArrayPreprocessor
from utilities.io import HDF5DatasetGenerator
from sklearn.metrics import classification_report
from keras.models import load_model
import json
import numpy as np

#brings in from the meanPreprocessor
means = json.loads(open(config.DATASET_MEAN).read())
mp  = MeanPreprocessor(means['R'], means['G'], means['B'])
iap = ImageToArrayPreprocessor(227, 227)
cp  = CropPreprocessor(227, 227)
model = load_model(config.MODEL_PATH)

#making a generator obhject
testGen = HDF5DatasetGenerator(config.TEST_HDF5, 32, preprocessors = [mp], classes = 2)
predictions = []
for (images, labels) in testGen.generator(passes = 1):
    for image in images:
        crops = cp.preprocess(image)
        crops = np.array([iap.preprocess(c) for c in crops], dtype = "float32")
        pred = model.predict(crops)
        predictions.append(pred.mean(axis = 0))        











