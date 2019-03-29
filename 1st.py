from glob import glob
files= glob('../project/dataset1/**/*', recursive='True')

extension=list()
for image in files:
    ext=image[-3:]
    if ext not in extension:
        extension.append(ext)
alpha_ext=list()
for ex in extension:
    if ex.isalpha()==True:
        alpha_ext.append(ex)
print(alpha_ext)

Data=glob("../project/dataset1/*/*/*.png", recursive=True)

from PIL import Image
from tqdm import tqdm

x=1
dimensions=list();
for images in tqdm(Data):
    dim= Image.open(images)
    size=dim.size
    if size not in dimensions:
        dimensions.append(size)
        x=x+1
    if(x>3):
        break
print(dimensions)

import os, sys, cv2
for images in Data:
    img=cv2.imread(images,1)
    name= images.split(".")
    re=cv2.resize(img, (50,50),3)
    cv2.imwrite(str(name[0]+".png"),img=re)

dimensions=list();
for images in tqdm(Data):
    dim= Image.open(images)
    size=dim.size
    if size not in dimensions:
        dimensions.append(size)
        x+=1
    if(x>3):
        break
print(dimensions)

from tqdm import tqdm
import csv
data_output=list()
data_output.append(["Classes"])
for file_name in tqdm(Data):
    data_output.append(file_name[-10:-4])

with open("output.csv", "w") as f:
    writer=csv.writer(f)
    for val in data_output:
        writer.writerows([val])

import pandas as pd
data_output = pd.read_csv("output.csv")

import cv2 #used for computer vision tasks such as reading image from file, changing color channels etc
import matplotlib.pyplot as plt #for plotting various graph, images etc.
def view_images(image): #function to view an image
    image_cv = cv2.imread(image) #reads an image
    plt.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)); #displays an image
view_images(Data[52])

import numpy as np
class1 = data_output[(data_output["Classes"] == 1)].shape[0]
class0 = data_output[(data_output["Classes"] == 0)].shape[0]
objects = ["class1", "class0"]
y_pos = np.arange(len(objects))
count = [class1, class0]
plt.bar(y_pos, count, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Number of images')
plt.xlabel("classes")
plt.title('Class distribution')
plt.show()

from sklearn.utils import shuffle
Data, data_output=shuffle(Data,data_output)

data_output=data_output.replace("class0",0)
data_output=data_output.replace("class1",1)

from keras.utils import to_categorical
data_output_encoded=to_categorical(data_output, num_classes=2)
print(data_output_encoded.shape)

from sklearn.model_selection import train_test_split
Data=np.array(Data)
x_train, x_test, y_train, y_test= train_test_split(Data, data_output_encoded, test_size=0.3)
print(Data.shape)
x_train=x_train[0:7000]
y_test=y_test[0:3000]
x_test=x_test[0:3000]
y_train=y_train[0:7000]

x_test=x_test.reshape(3000,1)
x_train=x_train.reshape(7000)

from keras.utils import to_categorical #to hot encode the data
from imblearn.over_sampling import RandomOverSampler

X_train_shape = x_train.shape[0]*x_train.shape[1]
X_test_shape = x_test.shape[0]*x_test.shape[1]

random_US = RandomOverSampler(ratio='auto') #Constructor of the class to perform undersampling
X_train_RUS, Y_train_RUS = random_US.fit_sample(x_train, y_train) #resamples the dataset
X_test_RUS, Y_test_RUS = random_US.fit_sample(x_test, y_test) #resamples the dataset

for i in range(len(X_train_RUS)):
    X_train_RUS_Reshaped = X_train_RUS.reshape(len(X_train_RUS),(50,50,3))

for i in range(len(X_test_RUS)):
    X_test_RUS_Reshaped=X_test_RUS.reshape(len(X_test_RUS),(50,50,3))

class1=1
class0=0

for i in range(0, len(Y_train_RUS)):
    if (Y_train_RUS[i] == 1):
        class1 += 1
for i in range(0, len(Y_train_RUS)):
    if (Y_train_RUS[i] == 0):
        class0 += 1
# For Plotting the distribution of classes
classes = ["class1", "class0"]
y_pos = np.arange(len(classes))
count = [class1, class0]
plt.bar(y_pos, count, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Number of images')
plt.title('Class distribution')
plt.show()

Y_train_encoded = to_categorical(Y_train_RUS, num_classes = 2)
Y_test_encoded = to_categorical(Y_test_RUS, num_classes = 2)

X_test, X_valid, Y_test, Y_valid = train_test_split(X_test_RUS, Y_test_encoded, test_size=0.2,shuffle=True)

print("Number of train files",len(X_train_RUS))
print("Number of valid files",len(X_valid))
print("Number of train_target files",len(Y_train_encoded))
print("Number of  valid_target  files",len(Y_valid))
print("Number of test files",len(X_test))
print("Number of  test_target  files",len(Y_test))

from sklearn.utils import shuffle
X_train,Y_train= shuffle(X_train_RUS,Y_train_encoded)

print(Y_train_encoded.shape)
print(Y_test.shape)
print(Y_valid.shape)

print("Training Data Shape:", X_train.shape)
print("Validation Data Shape:", X_valid.shape)
print("Testing Data Shape:", X_test.shape)
print("Training Label Data Shape:", Y_train.shape)
print("Validation Label Data Shape:", Y_valid.shape)
print("Testing Label Data Shape:", Y_test.shape)

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D #Import layers for the model
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential #Our model will be Sequential

model = Sequential()
model.add(Conv2D(32,3,3, kernel_size=(3,3), strides=(2,2), activation='relu',input_shape=(50,50,3)))
model.add(Flatten()) #Flattens the matrix into a vector
model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint  #Checkpoint to save the best weights of the model.
checkpointer = ModelCheckpoint(filepath='weights.best.cnn.hdf5',
                               verbose=1, save_best_only=True)
model.fit(X_train_RUS, Y_train_encoded,
          validation_data=(X_valid, Y_valid),
          epochs=3, batch_size=128, callbacks=[checkpointer], verbose=2,shuffle=True)
newpred=model.predict('idc.png')