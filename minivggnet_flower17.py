# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 14:10:16 2019

@author: Michael
"""

#importing the necessary directories
import os 
import shutil
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utilities.preprocessing import AspectAwarePreprocessor
from utilities.datasets import SimpleDatasetLoader
from keras.optimizers import SGD
from utilities.nn.cnn import MiniVGGNet
import matplotlib.pyplot as plt
import numpy as np
import imutils
from imutils import paths

#extracting the dataset
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

#building the baseline model
opt = SGD(lr=0.05, momentum=0.9, nesterov= True)
model = MiniVGGNet.build(width=64, height=64,depth = 3, classes = len(flower_className))
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
H = model.fit(trainX, trainY, validation_data = (testX, testY),
          batch_size=32, epochs = 100, verbose =1)

#evaluating the network
predictions = model.predict(testX, batch_size = 32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), 
                            target_names = flower_className ))

#plot the acciuracy network
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["acc"], label = "Training accuracy")
plt.plot(np.arange(0, 100), H.history["val_acc"], label = "Validation accuracy")
plt.tltle("Training loss and accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()



























