# -*- coding: utf-8 -*-
"""
Created on Sat May 18 16:41:25 2019

@author: H P ENVY
"""

from sklearn.feature_extraction.image import extract_patches_2d

class PatchPreprocessor:
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
    def preprocess(self, image):
        return extract_patches_2d(image, (self.height, self.width) ,max_patches = 1)[0]
    
    
    
    
    