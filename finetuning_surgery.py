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
from keras.applications import VGG16, inception_v3, vgg19
from keras.layers import Input
from keras.models import Model
from imutils import paths
import numpy as np
import os
import cv2

##prepare the image data generator for data augmentaion
#aug = ImageDataGenerator(rotation_range=30, width_shift_range = 0.1, height_shift_range = 0.1,
#                         shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, fill_mode = "nearest")

#preparing the data and the list of images
imagePaths = list(paths.list_images(r"C:\Users\H P ENVY\Desktop\Data Science\My directory set-up for Computer-Vision\datasets\Flowers17"))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]
#baseModel  = VGG16()
aap = AspectAwarePreprocessor(224,224)
isp = ImageToArrayPreprocessor()

#load the dataset from disk and then 
sdl = SimpleDatasetLoader(preprocessors = [aap, isp])
(data, labels) = sdl.load(imagePaths, verbose = 1)
data = data.astype("float") / 255.0

#split the dataset into the required stages
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.25, random_state = 42)

#Binarize the labels
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

#introduce the baseModel, buiid the headmodel, introduce the class objects
#baseModel = inception_v3.InceptionV3
baseModel = VGG16(weights = "imagenet", include_top = False, input_tensor  = Input(shape = (224,224,3)))
headModel = FCHeadNet.build(baseModel, len(classNames), 256)
model = Model(inputs  = baseModel.input , outputs = headModel)

#freezing the layers in the baseModel and warming them up
for layer in baseModel.layers:
    layer.trainable = False

#warming up the mdoel for some action
optimizer = RMSprop(lr = 0.001)
model.compile(loss = "categorical_crossentropy",metrics = ['accuracy'], optimizer = optimizer)
model.fit(trainX, trainY, batch_size = 16, epochs =100, validation_data = (testX, testY))

#predicting the classification result
preds = model.predict(testX, batch_size  = 32)
print(classification_report(testY.argmax(axis = 1), preds.argmax(axis = 1), target_names  = classNames))

#Now to the real classification
for layer in baseModel.layers:
    layer.trainable = True
    
opt = SGD(lr 0.01)
model.compile(loss = "categorical_crossentropy", metrics = ['accuracy'], optmizer = opt)
model.fit(trainX, trainY, batch_size = 16, epochs = 100, validation_data  = (testX, testY))








