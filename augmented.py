from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import glob

datagen = ImageDataGenerator(
    # rotation_range=90,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    # fill_mode="constant" , cval= 125
    fill_mode="nearest"
)
k = "data/train/*.*"
for file in glob.glob(k):
    img = tf.keras.preprocessing.image.load_img(file)
    test = tf.keras.preprocessing.image.img_to_array(img)
    test = test.reshape((1,) + test.shape)
    name = file.split("\\")[1]

    i = 1
    for batch in datagen.flow(test, batch_size=32,
        save_to_dir = "test", save_prefix= name, save_format="jpeg"):
        i += 1
        if i > 19:
            break
