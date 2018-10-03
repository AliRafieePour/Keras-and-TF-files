import numpy as np
import tensorflow as ts
import keras
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
batch_size = 128
num_classes = 10
epochs = 20

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784) 
X_test = X_test.reshape(10000, 784) 
X_train = X_train.astype('float32')
X_test = X_test.astype('float32') 
X_train /= 255
X_test /= 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

from keras import backend as K
#from keras.layers import Activationform 
from keras.utils.generic_utils import get_custom_objects

def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': swish})


model = Sequential()
model.add(Dense(1000, activation='relu', input_shape=(784,)))
model.add(Dropout(0.25))
model.add(Dense(1000, activation='swish'))
model.add(Dropout(0.25))
model.add(Dense(1000, activation='tanh'))
model.add(Dropout(0.25))
model.add(Dense(1000, activation='swish'))
model.add(Dropout(0.25))
model.add(Dense(1000, activation='swish'))
model.add(Dropout(0.25))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
