import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import pickle

Directory = r"C:\Users\wpf20\PycharmProjects\cnn\data"
Cate = ["pos","neg"]

IMG_SIZE = 300

data = []

for category in Cate:
    folder = os.path.join(Directory, category)
    label = Cate.index(category)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
        data.append([img_arr, label])
len(data)
random.shuffle(data)

x = []
y = []
for features, label in data:
    x.append(features)
    y.append(label)

x = np.array(x)
y = np.array(y)

pickle.dump(x, open("x.pkl", "wb"))
pickle.dump(y, open("y.pkl", "wb"))