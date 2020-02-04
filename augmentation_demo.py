# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 17:49:44 2019

@author: Michael
"""

#importing the necessary libraries
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import numpy as np

#choose image and produce many sets of images
image = load_img(r'C:\Users\Michael\Pictures/the_dog.jpg')
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

#the main augmentation ans saving in file
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
                         shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                         fill_mode="nearest")
imageGen = aug.flow(image, batch_size=1,save_to_dir=r'C:\Users\Michael\Desktop\Data Science\My directory set-up for Computer-Vision\Deep learning for computer vision - Practitioneers bundle',
                    save_prefix='imageDataGen images', save_format='jpg')

#using the generator object to generate relevant images
total = 0
for image in imageGen:
    total +=1
    
    if total == 10:
        break















