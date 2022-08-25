import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D
import os
import cv2, PIL as pillow
import random

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# make model if it doesnt exist already
if not os.path.exists('model.h5'):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10)

    model.save('model.h5')

# load model
loaded_model = tf.keras.models.load_model('model.h5')

rand_digit = x_test[random.randrange(0, x_test.shape[0])]
