from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.utils import np_utils
import tensorflow as tf
import numpy as np
from tensorflow import keras

# Loading Dataset

x_train = np.load('x_train.npy').astype(np.float32)
y_train = np.load('y_train.npy').astype(np.float32)
x_val = np.load('x_val.npy').astype(np.float32)
y_val = np.load('y_val.npy').astype(np.float32)

model = Sequential()
model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28,1)))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(320, activation='relu'))
model.add(keras.layers.Dense(100, activation='softmax'))
model.summary()

