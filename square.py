#!/usr/bin/python3

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import activations

model = keras.Sequential([keras.layers.Dense(units=1000, input_shape=[1], activation='sigmoid')])
model.add(keras.layers.Dense(1, activation='linear'))

model.compile(optimizer='sgd', loss='mean_squared_error')

xs=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ]         # input
ys=[0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100 ]  # expected output

model.fit(xs, ys, epochs=20000) # training


print(model.summary())

while(True):
    number = input("Number: ")
    n = float(number)
    o = model.predict([n])
    print("Result:", o)


