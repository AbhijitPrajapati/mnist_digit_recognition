import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D
import os
import pygame
from pygame import gfxdraw
from pyautogui import alert

import cv2, PIL as pillow

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

class App:
    def __init__(self):
        pygame.init()
        self.pixel_size = 35
        self.screen = pygame.display.set_mode((28 * self.pixel_size, 28 * self.pixel_size))
        pygame.display.set_caption('Draw a digit')
        self.clock = pygame.time.Clock()
        self.running = True
        self.screen.fill((0, 0, 0))
    
    def done(self):
        # convert to numpy array
        img = np.zeros((28, 28))
        for i in range(28):
            for j in range(28):
                img[j][i] = self.screen.get_at((i * self.pixel_size, j * self.pixel_size))[0]

        img = img.reshape(1, 28, 28, 1)

        # predict
        prediction = loaded_model.predict(img)
        alert(f'Prediction: {np.argmax(prediction[0])}', 'Prediction', button='OK')

        self.running = False

    def draw(self):
        pos = pygame.mouse.get_pos()
        x = pos[0] // self.pixel_size * self.pixel_size
        y = pos[1] // self.pixel_size * self.pixel_size
        #pygame.draw.rect(self.screen, (255, 255, 255), (x - self.pixel_size, y - self.pixel_size, self.pixel_size * 2, self.pixel_size * 2))
        '''
        points = [(x - self.pixel_size, y - self.pixel_size),
                (x + self.pixel_size, y - self.pixel_size),
                (x + self.pixel_size, y + self.pixel_size),
                (x - self.pixel_size, y + self.pixel_size)]
        '''
        points = [(x, y),
                (x + self.pixel_size, y),
                (x + self.pixel_size, y + self.pixel_size),
                (x, y + self.pixel_size)]
        gfxdraw.aapolygon(self.screen, points, (255, 255, 255))
        gfxdraw.filled_polygon(self.screen, points, (255, 255, 255))

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if pygame.mouse.get_pressed()[0]:
                    self.draw()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        self.done()

            self.clock.tick(300)
            pygame.display.update()

if __name__ == '__main__':
    App().run()
            