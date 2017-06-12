import numpy as np
import random

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adadelta, Adagrad

from keras.models import load_model
#model = load_model('model_myc_ffnn.h5')

f = open('gm12878_chr22_mod_npscore.dat')

l = f.readlines()

len_l = len(l)

for i in range(len_l):
    l[i] = l[i].split()

for i in range(len_l):
    l[i][0] = float(l[i][0])
    l[i][1] = float(l[i][1])
    l[i][2] = float(l[i][2])
    l[i][3] = float(l[i][3])
    l[i][4] = float(l[i][4])
    l[i][5] = float(l[i][5])

random.shuffle(l)

batch_size = 32
num_classes = 2
epochs = 10

dataset_split = 0.8

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
        x_train.append([[data[i][0], data[i][1], data[i][2], data[i][3], data[i][4]]])
        y_train.append(data[i][-1])
    
    for i in range(int(split * len_data), len_data):
        x_test.append([[data[i][0], data[i][1], data[i][2], data[i][3], data[i][4]]])
        y_test.append(data[i][-1])

    return (x_train, y_train), (x_test, y_test)


print('Loading data')
(x_train, y_train), (x_test, y_test) = create_training_and_test_set(l, split=dataset_split)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = x_train.reshape(int(dataset_split*len_l), 5)
x_test = x_test.reshape(len_l - int(dataset_split*len_l), 5)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)

#### Train the model ####

#'''
model = Sequential()
model.add(Dense(32, activation='relu', input_shape = (5, )))
model.add(Dropout(0.2))
model.add(Dense(12, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))

model.summary()

model.compile(loss='mean_squared_error',
              optimizer=Adagrad(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

model.save('model_myc_ffnn.h5')
f.close()
#'''
exit()
#### Training ends ####
#### Test the model ####

tp = 0 # True Pos
fp = 0 # False Pos
tn = 0 # True Neg
fn = 0 # False Neg

p = []
cnt_pos = 0
cnt_neg = 0
for i in range(len(x_test)):
    print(i)
    x_test[i] = np.array(x_test[i])
    x_test[i] = x_test[i].reshape(1, 5)
    x_test[i] = x_test[i].astype('float32')
    
    predicted = model.predict(x_test[i], verbose=0)
    predicted = int(predicted[0])
    
    actual = int(y_test[i])

    if (actual == 1) and (predicted == 1): # True Positive
        tp += 1
    if (actual == 0) and (predicted == 1): # False Positive
        fp += 1
    if (actual == 1) and (predicted == 0): # False Negative
        fn += 1
    if (actual == 0) and (predicted == 0): # True Negative
        tn += 1
    
    
    #if (np.array_equal(np.array([0, 1]).reshape(1, 2).astype('float32'), predicted)):
    #    cnt_pos += 1
    #elif (np.array_equal(np.array([1, 0]).reshape(1, 2).astype('float32'), predicted)):
    #    cnt_neg += 1
        
    #if np.array_equal(output_cases['01'], op):
    #    print np.array_equal(actual, predicted)

precision = recall = -1
try:
    precision = float(tp) / (tp + fp)
except ZeroDivisionError as z:
    pass

try:
    recall = float(tp) / (tp + fn)
except ZeroDivisionError as z:
    pass

print('Precision:', precision)
print('Recall:', recall)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

f.close()
