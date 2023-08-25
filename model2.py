import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__

import numpy as np
from keras.preprocessing import image
from keras.utils import load_img , img_to_array

arr = []

def prediction(train , test , predict):
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = False)

    test_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = False)

    training_set1 = train_datagen.flow_from_directory(train,
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

    test_set1 = test_datagen.flow_from_directory(test,
                                            target_size = (64 , 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

    cnn1 = tf.keras.models.Sequential()
    cnn1.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
    cnn1.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn1.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    cnn1.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn1.add(tf.keras.layers.Flatten())
    cnn1.add(tf.keras.layers.Dense(units=128, activation='relu'))
   
    cnn1.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    cnn1.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    cnn1.fit(x = training_set1, validation_data = test_set1, epochs = 20)

    test_image1 = load_img(predict, target_size = (64 , 64))
    test_image1 = img_to_array(test_image1)
    test_image1 = np.expand_dims(test_image1, axis = 0)
    result = cnn1.predict(test_image1)

    return result[0][0]


train2 = (r'D:\PBL\lettersize')
test2 = (r'D:\PBL\testlettersize')
predict1 = (r'D:\PBL\testset\test3.jpeg')

result1 = prediction(train2 , test2 , predict1)

if result1 == 1:
    arr.append("You are Introvert")
else:
    arr.append("You are Extrovert")

train1 = (r'D:\PBL\pressure')
test1 = (r'D:\PBL\testpressure')
predict2 = (r'D:\PBL\testset\test1.jpeg')

result2 = prediction(train1 , test1 , predict2)

if result2 == 0:
    arr.append("You are Calm")
else:
    arr.append("You React Quickly")

train3 = (r'D:\PBL\slant')
test3 = (r'D:\PBL\testslant')
predict3 = (r'D:\PBL\testset\test3.jpeg')

result3 = prediction(train3 , test3 , predict3)

if result3 == 1:
    arr.append("You are Balanced")
else:
    arr.append("You are Optimistic")

print(arr)

