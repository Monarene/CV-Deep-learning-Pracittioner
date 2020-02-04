# -*- coding: utf-8 -*-
"""
Created on Sat May 18 16:26:53 2019

@author: H P ENVY
"""

import cv2

class MeanPreprocessor:
    
    def __init__(self, rMean, gMean, bMean):
        self.rMean  = rMean
        self.gMean  = gMean
        self.bMean = bMean
        
    def preprocess(self, image):
        B, G, R = cv2.split(image.astype("float32"))
        
        R -= self.rMean
        G -= self.gMean
        B -= self.bMean
        
        return cv2.merge([B, G, R])
        