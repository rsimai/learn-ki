#!/usr/bin/python3

import tensorflow as tf
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
#model.add(tf.keras.layers.Dense(1))


model.compile(optimizer='sgd', loss='mean_squared_error')

xs=[0, 1, 2, 3, 4]         # input
ys=[1, 2, 3, 4, 5]       # expected output

model.fit(xs, ys, epochs=1000) # training

#print("1 weights:", model.layers[0].get_weights()[0])
#print("1 biases: ", model.layers[0].get_weights()[1])
#print("2 weights:", model.layers[1].get_weights()[0])
#print("2 biases: ", model.layers[1].get_weights()[1])

print(model.summary())

while(True):
    number = input("Number: ")
    n = float(number)
    o = float(model.predict([n]))
    print("Result:", o)


