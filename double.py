#!/usr/bin/python3

import tensorflow as tf
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

xs=[0, 1, 2, 3, 4]         # input
ys=[0, 2, 4, 6, 8]       # expected output

model.fit(xs, ys, epochs=1000) # training

print(model.summary())

while(True):
    number = input("Number: ")
    n = float(number)
    o = float(model.predict([n]))
    print("Result:", o)


