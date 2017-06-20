import numpy as np
import random
import sys
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras import regularizers
from keras.optimizers import *
from keras.models import load_model
f = open('peak_68pct.csv')

l = f.readlines()
random.shuffle(l)
random.shuffle(l)
random.shuffle(l)
f.close()
print 'starting'
random.shuffle(l)

len_l = len(l)

for i in range(len_l):
    l[i] = l[i].split(',')

for i in range(len_l):
    l[i][0] = float(l[i][0])
    l[i][1] = float(l[i][1])
    l[i][2] = float(l[i][2])
    #l[i][3] = float(l[i][3])
    if l[i][3] == 'no\n':
        l[i][3] = 0.0
    else:
        l[i][3] = 1.0

batch_size = 256
num_classes = 2
epochs = 4

dataset_split = 0.7

def create_training_and_test_set(data, split=0.6):
    '''
        split: float, percentage to split
        returns: (x_train, y_train), (x_test, y_test)
    '''
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    
    len_data = len(data)
    
    for i in range(int(split * len_data)):
        #x_train.append([[data[i][0], data[i][1], data[i][2], data[i][3], data[i][4]]])
        #y_train.append(data[i][-1])
        x_train.append([[data[i][0], data[i][1], data[i][2]]])
        y_train.append(data[i][3])

        
    for i in range(int(split * len_data), len_data):
        #x_test.append([[data[i][0], data[i][1], data[i][2], data[i][3], data[i][4]]])
        #y_test.append(data[i][-1])
        x_test.append([[data[i][0], data[i][1], data[i][2]]])
        y_test.append(data[i][3])

        
    return (x_train, y_train), (x_test, y_test)


print('Loading data')
(x_train, y_train), (x_test, y_test) = create_training_and_test_set(l, split=dataset_split)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)


x_train = x_train.reshape(int(dataset_split*len_l), 1, 3)
x_test = x_test.reshape(len_l - int(dataset_split*len_l), 1, 3)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()
model.add(LSTM(32, activation='relu', input_shape = (1, 3), kernel_regularizer=regularizers.l2(0.01), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('model_myc.h5')
f.close()
