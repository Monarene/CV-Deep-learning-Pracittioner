# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 14:39:31 2018

@author: Michael
"""

#importing te relevant libraries
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation
from keras.layers.convolutional import Conv2D 
from keras import backend as K

#defining the shallownet class
class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        
        model = Sequential()
        input_shape = (height, width, depth)
        if K.image_data_format == "channels_first":
            input_shape = (depth, height, width)
            
        #now defining the network architecture
        model.add(Conv2D(32, (3,3), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        
        return model
        
        