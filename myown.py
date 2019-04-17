import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Convolution2D, Dense, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
from keras.models import load_model

import os, sys, cv2
from glob import glob

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True
                                   )

test_data=ImageDataGenerator(rescale=1./255)

training_set= train_datagen.flow_from_directory('C:/Users/karti/Desktop/project/breastcancer/train/'
                                                ,
                                                target_size=(50,50),

                                                class_mode='categorical',
                                                )
test_set= test_data.flow_from_directory('C:/Users/karti/Desktop/project/breastcancer/test/',
                                        target_size=(50,50),

                                        class_mode='categorical'
                                        )
from sklearn.utils import shuffle

training_set.filenames, training_set.classes=shuffle(training_set.filenames, training_set.classes)

test_set.filenames, test_set.classes=shuffle(test_set.filenames, test_set.classes)

classifier=Sequential()
classifier.add(Convolution2D(32, 3, 3, input_shape=(50,50,3), activation="relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))

classifier.add(Flatten())

classifier.add(Dense(output_dim = 128, activation = 'relu'))

classifier.add(Dense(output_dim=128, activation='relu'))

classifier.add(Dense(output_dim = 128, activation = 'relu'))

classifier.add(Dropout(0.10))

classifier.add(Dense(output_dim=128, activation='relu'))

classifier.add(Dropout(0.20))

classifier.add(Dense(output_dim = 2, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

print(classifier.summary())

classifier.fit_generator(training_set,
                         samples_per_epoch= 1500,
                         nb_epoch = 10 ,
                         validation_data = test_set,
                         nb_val_samples=len(test_set.filenames))

classifier.save('classifier11.h5')
from keras.models import load_model
from keras.preprocessing import image

import numpy as np
from keras_preprocessing import image
# dimensions
img_width, img_height = 50,50
# predicting images
img = image.load_img('idc.png', target_size=(img_width, img_height))
img= image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)
classes = classifier.predict(img)
print(classes)