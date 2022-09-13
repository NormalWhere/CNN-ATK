from tensorflow import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.optimizers import RMSprop
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import keras
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Dropout ,Activation, Flatten
from keras.layers import Conv2D, MaxPool2D
from os import listdir
from os.path import isfile, join

width = 128
num_classes = 2
train_path = "Data/train/"
test_path = "Data/test/"
trainImg = [train_path+f for f in listdir(train_path) if listdir(join(train_path, f))]
testImg = [test_path+f for f in listdir(test_path) if listdir(join(test_path, f))]


def img2data(path):
    rawImgs = []
    labels = []

    for image_Path in (path):
        for item in tqdm(listdir(image_Path)):
            file = join(image_Path, item)
            if file[-1] == 'g':
                img = cv2.imread(file, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (width, width))
                rawImgs.append(img)

                l = image_Path.split('/')[1]

                if l == 'Neg':
                    labels.append([1, 0])
                elif l == 'Pos':
                    labels.append([0, 1])

    return rawImgs, labels

x_train, y_train = img2data(trainImg)
x_test, y_test = img2data(testImg)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x_train.shape,y_train.shape,x_test.shape, y_test.shape

model0 = Sequential([
        Conv2D(128, (3,3), activation='relu', input_shape=(width, width, 3)),
        MaxPool2D(2),
        Conv2D(128,(3,3) , activation='relu'),
        MaxPool2D(pool_size=(2,2 )),
        Dense(16),
        Flatten(),

        Dense(num_classes, activation='softmax')
    ])

model0.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics= ['accuracy'])
batch_size = 32
epochs = 10

history = model0.fit(x_train, y_train ,batch_size=batch_size, epochs=epochs ,validation_data=(x_test, y_test))

testpath = 'test/'
testImg = [testpath+f for f in listdir(testpath)]
for imagePath in (testImg):
    for i,item in enumerate(listdir(imagePath)):
        file = join(imagePath, item)
        print(file)

testpath = 'test/'
testImg = [testpath + f for f in listdir(testpath)]
rimg = []
for imagePath in (testImg):
    for i, item in enumerate(listdir(imagePath)):

        file = join(imagePath, item)
        if file[-1] == 'g':
            print(file)
            imgori = cv2.imread(file)
            imgori = cv2.cvtColor(imgori, cv2.COLOR_BGR2RGB)
            img = cv2.resize(imgori, (width, width))
            rimg = np.array(img)
            rimg = rimg.astype('float32')
            rimg /= 255
            rimg = np.reshape(rimg, (1, 128, 128, 3))
            predict = model0.predict(rimg)
            label = ['Pos', 'Neg']
            result = label[np.argmax(predict)]
            plt.title(imagePath)
            cv2.putText(imgori, str(result), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            plt.imshow(imgori)
            plt.show()