from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf

conv1 = tf.keras.Sequential()
conv1.add(Conv2D(10,(3,3), activation='relu', padding='same', input_shape(28,28,1)))
conv1.add(MaxPooling2D((2,2)))
conv1.add(Flatten())
conv1.add(Dense(100, activation='relu'))
conv1.add(Dense(10, activation='softmax'))
conv1.compile(optimizer='adam', loss='catergorical_crossentropy', metrics=['accuracy'])

# Loading Dataset

x_train = np.load('dataset/x_train.npy').astype(np.float32)
y_train = np.load('dataset/y_train.npy').astype(np.float32)
x_val = np.load('dataset/x_val.npy').astype(np.float32)
y_val = np.load('dataset/y_val.npy').astype(np.float32)
