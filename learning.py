#!/usr/bin/python

import tensorflow as tf
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

xs=[1, 3, 5]
ys=[2, 6, 10]

model.fit(xs, ys, epochs=1000)

while(True):
    number = input("Number: ")
    n = float(number)
    print("Result:", model.predict([n]))


