from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


boston_housing = keras.datasets.boston_housing

(train_data,train_labels), (test_data,test_labels) = boston_housing.load_data()

order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

#normalize features

mean = train_data.mean(axis = 0)
std = train_data.std(axis = 0)
train_data = (train_data - mean)/std
test_data = (test_data - mean)/std

print(train_data[0])

def build_model():
    model = tf.contrib.eager.Sequential([
        tf.layers.Dense(64, activation=tf.nn.relu,input_shape=(train_data.shape[1],)),
        tf.layers.Dense(64, activation=tf.nn.relu),
        tf.contrib.layers.fully_connected()
    ])

    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse',optimizer=optimizer,metrics=['rmse'])
    return model

def plot_history(history):
    plt.figure()
    plt.xlabel('EPOCH')
    plt.ylabel('Mean Abs Error')
    plt.plot(history.epoch, np.array(history.history['mean abs error']), label = 'Train loss')
    plt.plot(history.epoch, np.array(history.history['val_mean abs error']), label = 'Val loss')
    plt.legend()
    plt.ylim([0,5])


model = build_model()
model.summary()

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0: print('')
        print('.', end ='')

EPOCHS = 500

logs = model.fit(train_data, train_labels, epochs = EPOCHS, validation_split =0.2, verbose=0, callbacks = [PrintDot()])

plot_history(logs)
