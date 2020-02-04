# -*- coding: utf-8 -*-
"""
Created on Fri May 10 22:57:36 2019

@author: H P ENVY
"""

from keras.layers.core import Dropout, Flatten, Dense

class FCHeadNet:
    @staticmethod
    def build(baseModel, classes, D):
        headModel = baseModel.output
        headModel = Flatten(name = "flatten")(headModel)
        headModel = Dense(D, activation = "relu")(headModel)
        headModel = Dense(512, activation = "relu")(headModel)
        headModel  = Dropout(0.5)(headModel)
        
        #adding the softmax layer
        headModel  = Dense(classes, activation = "softmax")(headModel)
        return headModel
        