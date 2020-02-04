# -*- coding: utf-8 -*-
"""
Created on Tue May 21 20:32:16 2019

@author: H P ENVY
"""

#importing the necessary libraries
import matplotlib
matplotlib.use("Agg")
from config import dogs_vs_cats_config as config
from utilities.callbacks import TrainingMonitor
from utilities.preprocessing import MeanPreprocessor, SimplePreprocessor, ImageToArrayPreprocessor, PatchPreprocessor
#from utilities
from utilities.io import HDF5DatasetGenerator
from utilities.nn.cnn import AlexNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import os
import json

#constructing the image generator and activating the preprocessors
means = json.loads(open(config.DATASET_MEAN).read())
aug = ImageDataGenerator(rotation_range = 20, zoom_range = 0.15, width_shift_range = 0.2, height_shift_range = 0.2,
                         shear_range = 0.15, horizontal_flip = True, fill_mode = "nearest" )
iap = ImageToArrayPreprocessor()
pp = PatchPreprocessor(227, 227)
sp = SimplePreprocessor(227, 227) 
mp = MeanPreprocessor(means["R"], means["G"], means["B"])

#initialize the dataset generators
trainGen  = HDF5DatasetGenerator(config.TRAIN_HDF5, 32, aug=aug, preprocessors =[pp, mp, iap], classes = 2)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 32, aug=aug, preprocessors=[sp, mp, iap], classes = 2)

#compile the model and do al the deep learning
optimizer  = Adam(lr = 1e-3)
model = AlexNet.build(width = 227, height = 227, depth = 3, reg = 0.0002, classes = 2)
model.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics = ["accuracy"])

#construct callbacks
path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(os.getpid())])
callbacks = [TrainingMonitor(path)]
model.fit_generator(trainGen.generator(), steps_per_epoch = trainGen.numImages // 32,
                   validation_data = valGen.generator(), validation_steps = valGen.numImages // 32,
                   epochs = 3, max_queue_size = 32*2, verbose = 1)
model.save(config.MODEL_PATH, overwrite = True)

trainGen.close()
valGen.close()

#it is very possible to access the all the HDF5 datasets generated, either through their generator or directly 
#through their hdf5 files
































