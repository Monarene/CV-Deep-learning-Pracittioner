# -*- coding: utf-8 -*-
"""
Created on Mon May 20 09:24:27 2019

@author: H P ENVY
"""

#importing the necessary commands
from keras.utils import np_utils
import h5py
import numpy as np

#building the main class
class HDF5DatasetGenerator:
    
    def __init__(self, dbPath, batchSize, aug=None, 
                 preprocessors = None, binarize = True, classes = 2):
        self.db = h5py.File(dbPath)
        self.batchSize = batchSize
        self.aug = aug
        self.binarize = binarize
        self.classes = classes
        self.preprocessors = preprocessors
        self.numImages = self.db["labels"].shape[0]
        
    def generator(self, passes = np.inf):
        epochs  = 0
        while epochs < passes:
            for i in np.arange(0, self.numImages, self.batchSize):
                images = self.db["images"][i:i + self.batchSize]
                labels = self.db["labels"][i:i  + self.batchSize]
                
                if self.binarize:
                    labels = np_utils.to_categorical(labels, self.classes)
                    
                if self.preprocessors is not None:
                    procImages = []
                    for image in images:
                        
                        for p in self.preprocessors:
                            image = p.preprocess(image)
                            
                        procImages.append(image)
                    images = np.array(procImages)
                
                if self.aug is not None:
                    (images, labels) = next(self.aug.flow(images, labels, batch_size = self.batchSize))
                    
                yield (images, labels)
                
            epochs +=1
            
    def close(self):
        self.db.close()
                
            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            