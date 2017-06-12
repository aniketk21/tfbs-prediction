import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.optimizers import Adam

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

batch_size = 32
num_classes = 2
epochs = 10

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

for i in range(len(y_test)):
    if y_test[i] == 1:
        print x_test[i], y_test[i]
        break

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

print x_train[0]
#x_train = x_train.reshape(int(dataset_split*len_l), 1, 5)
#x_test = x_test.reshape(len_l - int(dataset_split*len_l), 1, 5)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print x_train[0]

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()
model.add(LSTM(12, activation='relu', input_shape = (1, 5), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(12, activation='relu', return_sequences=False))
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
