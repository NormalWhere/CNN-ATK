from tflearn.data_utils import image_preloader
from PIL import Image
import numpy as np
import tensorflow as tf
import tflearn
import os
import random
import math
import matplotlib.pyplot as plt

IMAGE_FLODER = "data/train"
TRAIN_DATA = "data/training_data.txt"
TEST_DATA = "data/test_data.txt"
VALIDATION_DATA = "data/validation.txt"
train_proportion = 0.7
test_proportion = 0.2
validation_proportion = 0.1

filenames_image = os.listdir(IMAGE_FLODER)
random.shuffle(filenames_image)

total = len(filenames_image)

fr = open(TRAIN_DATA, "w")
train_files = filenames_image[0 :int(train_proportion*total)]
for filename in train_files:
    if filename[0:3] == "neg":
        fr.write(IMAGE_FLODER + '/' + filename + ' 0\n')
    elif filename[0:3] == "pos":
        fr.write(IMAGE_FLODER + '/' + filename + ' 1\n')
fr.close()

fr = open(TEST_DATA,'w')
test_file = filenames_image[int(math.ceil(train_proportion*total)) :int(math.ceil(train_proportion+test_proportion)*total)]
for filename in test_file:
    if filename[0:3] == "neg":
        fr.write(IMAGE_FLODER + '/' + filename + ' 0\n')
    elif filename[0:3] == 'pos':
        fr.write(IMAGE_FLODER + '/' + filename + ' 1\n')
fr.close()

fr = open(VALIDATION_DATA,'w')
valid_file = filenames_image[int(math.ceil((train_proportion+test_proportion)*total)):total]
for filename in valid_file:
    if filename[0:3] == "neg":
        fr.write(IMAGE_FLODER + '/' + filename + ' 0\n')
    elif filename[0:3] == 'pos':
        fr.write(IMAGE_FLODER + '/' + filename + ' 1\n')
fr.close()

x_train, y_train = image_preloader(TRAIN_DATA, image_shape=(70, 160), mode='file', categorical_labels=True, normalize=True)
x_test, y_test = image_preloader(TEST_DATA, image_shape=(70, 160), mode='file', categorical_labels=True, normalize=True)
x_val, y_val = image_preloader(VALIDATION_DATA, image_shape=(70, 160), mode='file', categorical_labels=True, normalize=True)

print("Dataset")
print("train {}".format(len(x_train)))
print("test {}".format(len(x_test)))
print("validation {}".format(len(x_val)))
print("shape {}".format(x_train[1].shape))
print("label:{}, class:{}".format(y_train[1].shape, len(y_train[1])))



x = tf.compat.v1.placeholder(tf.float32,shape=[None,70,160,3] , name='input_image')
#input class
y_ = tf.compat.v1.placeholder(tf.float32,shape=[None, 2] , name='input_class')
input_layer=x
#convolutional layer 1 --convolution+RELU activation
conv_layer1=tflearn.layers.conv.conv_2d(input_layer, nb_filter=64, filter_size=5, strides=[1,1,1,1],
                                        padding='same', activation='relu', regularizer="L2", name='conv_layer_1')

#2x2 max pooling layer
out_layer1=tflearn.layers.conv.max_pool_2d(conv_layer1, 2)


#second convolutional layer
conv_layer2=tflearn.layers.conv.conv_2d(out_layer1, nb_filter=128, filter_size=5, strides=[1,1,1,1],
                                        padding='same', activation='relu',  regularizer="L2", name='conv_layer_2')
out_layer2=tflearn.layers.conv.max_pool_2d(conv_layer2, 2)
# third convolutional layer
conv_layer3=tflearn.layers.conv.conv_2d(out_layer2, nb_filter=128, filter_size=5, strides=[1,1,1,1],
                                        padding='same', activation='relu',  regularizer="L2", name='conv_layer_2')
out_layer3=tflearn.layers.conv.max_pool_2d(conv_layer3, 2)

#fully connected layer1
fcl= tflearn.layers.core.fully_connected(out_layer3, 4096, activation='relu' , name='FCL-1')
fcl_dropout_1 = tflearn.layers.core.dropout(fcl, 0.8)
#fully connected layer2
fc2= tflearn.layers.core.fully_connected(fcl_dropout_1, 4096, activation='relu' , name='FCL-2')
fcl_dropout_2 = tflearn.layers.core.dropout(fc2, 0.8)
#softmax layer output
y_predicted = tflearn.layers.core.fully_connected(fcl_dropout_2, 2, activation='softmax', name='output')

#loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.math.log(y_predicted+np.exp(-10))))
#optimiser -
train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#calculating accuracy of our model
correct_prediction = tf.equal(tf.argmax(y_predicted,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# session parameters
sess = tf.compat.v1.InteractiveSession()
#initialising variables
init = tf.compat.v1.global_variables_initializer()
sess.run(init)
saver = tf.compat.v1.train.Saver()
save_path ="C:/Users/wpf20/PycharmProjects/cnn/mark1.ckpt"
# grabbing the default graph
g = tf.compat.v1.get_default_graph()

# every operations in our graph
[op.name for op in g.get_operations()]

epoch=3 # run for more iterations according your hardware's power
#change batch size according to your hardware's power. For GPU's use batch size in powers of 2 like 2,4,8,16...
batch_size=20
no_itr_per_epoch=len(x_train)//batch_size

no_itr_per_epoch
n_test = len(x_test)  # number of test samples
n_val = len(x_val)  # number of validation samples
# Now iterate over our dataset n_epoch times
for iteration in range(epoch):
    print("Iteration no: {} ".format(iteration))

    previous_batch = 0
    # Do our mini batches:
    for i in range(no_itr_per_epoch):
        current_batch = previous_batch + batch_size
        x_input = x_train[previous_batch:current_batch]
        x_images = np.reshape(x_input, [batch_size, 70, 160, 3])

        y_input = y_train[previous_batch:current_batch]
        y_label = np.reshape(y_input, [batch_size, 2])
        previous_batch = previous_batch + batch_size

        _, loss = sess.run([train_step, cross_entropy], feed_dict = {x: x_images, y_: y_label})
        if i % 100 == 0:
            print("Training loss : {}".format(loss))

    x_test_images = np.reshape(x_test[0:n_test], [n_test, 70, 160, 3])
    y_test_labels = np.reshape(y_test[0:n_test], [n_test, 2])
    Accuracy_test = sess.run(accuracy,
                             feed_dict={
                                 x: x_test_images,
                                 y_: y_test_labels
                             })
    Accuracy_test = round(Accuracy_test * 100, 2)

    x_val_images = np.reshape(x_val[0:n_val], [n_val, 70, 160, 3])
    y_val_labels = np.reshape(y_val[0:n_val], [n_val, 2])
    Accuracy_val = sess.run(accuracy,
                            feed_dict={
                                x: x_val_images,
                                y_: y_val_labels
                            })
    Accuracy_val = round(Accuracy_val * 100, 2)
    print("Accuracy ::  Test_set {} % , Validation_set {} % ".format(Accuracy_test, Accuracy_val))


def process_img(img):
    img = img.resize((70, 160))
    img = np.array(img)
    img = img / np.max(img).astype(float)
    img = np.reshape(img, [1, 70, 160, 3])
    return img

# test images
ex_image = Image.open('data/test/test.1.jpg')
test_image = process_img(ex_image)
predicted_array = sess.run(y_predicted, feed_dict={x: test_image})
predicted_class = np.argmax(predicted_array)
if predicted_class == 0:
    print("Negative")
else:
    print("Positive")
plt.imshow(ex_image)
plt.show()