# -*- coding: utf-8 -*-
"""
Created on Sat May 18 16:48:08 2019

@author: H P ENVY
"""

#importing the relevant classes
import numpy as np
import cv2

class CropPreprocessor:
    
    def __init__(self, width, height, inter = cv2.INTER_AREA, horiz = True):
        self.width = width
        self.height = height
        self.inter  = inter
        self.horiz  = horiz
        
    del preprocess(self, image):
        crops = []
        (h, w)  = image.shape[:2]
        coords = [
                [0, 0, self.width, self.height],
                [w - self.width, 0 ,w, self.height],
                [w - self.width, h - self.height, w, h],
                [0, h - self.height, w - self.width, h]]
        
        #compute the center crop of the images
        dW = int(0.5 * (w - self.width))
        dH = int(0.5 * (h - self.height))
        coords.append([dH, dW, w - dW, h - dH])
        
        for (startX, startY, endX, endY) in coords:
            crop  = image[startY:endY, startX:endX]
            crop = cv2.resize(crop, (self.width, self.height))
        
        
        
        
        
        
        