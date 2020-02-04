# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 02:46:22 2019

@author: Michael
"""

#importing the necessary libraries
import os
import shutil
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from utilities.preprocessing import AspectAwarePreprocessor
from utilities.datasets import SimpleDatasetLoader
from utilities.nn.cnn import MiniVGGNet

#importing the dataset
segment = 80
path = r'C:\Users\Michael\Desktop\Data Science\My directory set-up for Computer-Vision\Deep learning for computer vision - Practitioneers bundle\datasets\Flowers17'

pl = os.listdir(path)

flower_className = ['Daffodil', 'Snowdrop', 'Lily_Valley', 'Bluebell',
                    'Crocus', 'Iris', 'Tigerlily', 'Tulip',
                    'Fritillary', 'Sunflower', 'Daisy', 'Colts\'s_Foot',
                    'Dandelion', 'Cowslip', 'Buttercup', 'Windflower', 'Pansy']

for p in pl:
    if '.jpg' in p:
        index = int(p.split("_")[-1].strip(".jpg")) - 1
        classname = index // 80
        classname = flower_className[classname]
        os.makedirs(path + '/' + classname, exist_ok=True)
        shutil.move(path + '/' + p, path + '/' + classname + '/' + p)

print("[INFO]")
imagePaths = list(paths.list_images(r'C:\Users\Michael\Desktop\Data Science\My directory set-up for Computer-Vision\Deep learning for computer vision - Practitioneers bundle\datasets\Flowers17'))
aap = AspectAwarePreprocessor(64,64)
sdl = SimpleDatasetLoader(preprocessors=[aap])
(data, labels) = sdl.load(imagePaths, verbose=500)

#preprocessing the data
data = data.astype("float")/255.0
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
trainX, testX, trainY, testY = train_test_split(data, labels, random_state=42, test_size=0.25)

#building the netwok and applying data augmentaion
opt = SGD(lr = 0.05, nesterov=True, momentum = 0.9)
aug = ImageDataGenerator(rotation_range = 30, width_shift_range = 0.1, zoom_range = 0.2,
                         height_shift_range = 0.1, shear_range = 0.2, horizontal_flip = True,
                         fill_mode = "nearest")
model = MiniVGGNet.build(width = 64, height = 64, depth = 3, classes = len(flower_className))
model.compile(optimizer = opt, loss = "categorical_crossentropy", metrics = ["accuracy"])
H = model.fit_generator(aug.flow(trainX, trainY, batch_size = 32), steps_per_epoch = len(trainX)//32,
                    validation_data = (testX, testY), epochs = 100, verbose = 1)

#saving the model
model.save("MiniVGGNet on flowers 17 dataset with data augmentation.hdf5")

#plotting and evaluating the dataset progress reports
plt.style.use("ggplot")
plt.figure("MiniVGGNet on flowers 17 with data aumentation")
plt.plot(np.arange(0, 100), H.history["acc"], label = "Training accuracy")
plt.plot(np.arange(0, 100), H.history["val_acc"], label = "Validation accuracy")
plt.title("Training loss and accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("MiniVGGNet on flowers 17 with data aumentation")


















