# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:57:33 2019

@author: H P ENVY
"""

#importing the necessary libraries
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.regularizers import l2
from keras.models import Sequential
from keras import backend as K

#building the required class
class AlexNet:
    @staticmethod
    def build(height, width, depth, reg = 0.0002, classes = 2):
        input_shape = (height, width, depth)
        chanDim  = -1
        
        if K.image_data_format == "channels_first":
            input_shape = (depth, height, width)
            chanDim = 1
            
        #building the Convolutional Neural Network
        model  = Sequential()
        model.add(Conv2D(96, (11,11), strides = (4,4),input_shape = input_shape, padding="same", kernel_regularizer = l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (3,3), strides=(2,2)))
        model.add(Dropout(0.25))        
        
        model.add(Conv2D(256,(5,5), padding = "same", kernel_regularizer = l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis =chanDim))
        model.add(MaxPooling2D(pool_size = (3,3), strides = (2,2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(384,(3,3), padding = "same", kernel_regularizer = l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis =chanDim))
        
        model.add(Conv2D(384,(3,3), padding = "same", kernel_regularizer = l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))
       
        model.add(Conv2D(256,(3,3), padding = "same", kernel_regularizer = l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis =chanDim))
        model.add(MaxPooling2D(pool_size = (3,3), strides = (2,2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(4096, kernel_regularizer  = l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))
        model.add(Dropout(0.5))
        
        model.add(Dense(4096, kernel_regularizer  = l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))
        model.add(Dropout(0.5))
        
        model.add(Dense(classes, kernel_regularizer  = l2(reg)))
        model.add(Activation("softmax"))
        
        return model
        
        
        
        
        
        
        
        
        
        