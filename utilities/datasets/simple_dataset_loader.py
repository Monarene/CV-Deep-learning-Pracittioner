# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 21:46:10 2018

@author: Michael
"""

#importing the relevant libraries
import cv2
import numpy as np
import os

#building the class for 

class SimpleDatasetLoader:
    
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors
        if self.preprocessors == None:
            self.preprocessors = []
            
    def load(self, imagePaths, verbose=-1):
        data = []
        labels = []
        
        for (i, imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)
            data.append(image)
            labels.append(label)
        
        if verbose > 0 and i > 0 and ( i + 1) % verbose == 0:
            print("INFO processed {}/{}".format(i + 1, len(imagePaths)))
        
        return (np.array(data), np.array(labels))
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            