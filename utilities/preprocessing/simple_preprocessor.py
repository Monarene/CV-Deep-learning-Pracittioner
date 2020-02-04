# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 20:07:43 2018

@author: Michael
"""

#importing the necesary liabraries
import cv2
import numpy as np

#build processor class
class SimplePreprocessor:
    
    def __init__(self, width, height, inter = cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter
        
    def preprocess(self, image):
        return cv2.resize(image, (self.width, self.height),
                          interpolation = self.inter)