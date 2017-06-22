import numpy as np
import random
import sys

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import *
from keras import regularizers
from keras.models import load_model
import keras.backend as K

custom_metrics_flag = False 
to_file = False

def precision(y_true, y_pred):		
    """
        Precision metric.		
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))		
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))		
    precision = true_positives / (predicted_positives + K.epsilon())		
    return precision
def recall(y_true, y_pred):		
    """
        Recall metric.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

print 'Opening file...'
f = open('peak_68pct.csv')
l = f.readlines()
f.close()

print 'starting'
random.shuffle(l)
random.shuffle(l)
random.shuffle(l)
random.shuffle(l)

len_l = len(l)
for i in range(len_l):
    #print i
    l[i] = l[i].split(',')

for i in range(len_l):
    l[i][0] = float(l[i][0])
    l[i][1] = float(l[i][1])
    l[i][2] = float(l[i][2])
    if l[i][3] == 'no\n':
        l[i][3] = 0.0
    else:
        l[i][3] = 1.0

batch_size = 128
num_classes = 2
epochs = 5

threshold = 0.5 # threshold for the classifier
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
        x_train.append([[data[i][0], data[i][1], data[i][2]]])
        y_train.append(data[i][3])
    
    for i in range(int(split * len_data), len_data):
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

x_train = x_train.reshape(int(dataset_split*len_l), 3)
if not(custom_metrics_flag):
    x_test = x_test.reshape(len_l - int(dataset_split*len_l), 3)

x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
x_test = x_test.astype('float32')
y_test = y_test.astype('float32')

def custom_metrics(model, x_test, y_test, len_l, threshold, dataset_split, write_to_file=False):
    tp = 0 # True Pos
    fp = 0 # False Pos
    tn = 0 # True Neg
    fn = 0 # False Neg
    
    # write (precision, recall) to a file?
    if write_to_file:
        result = ''
    
    for i in xrange(len(x_test)):
        if not(i%100000):
            print(str(float(i)/(len_l - dataset_split*len_l)) + '%')
            print('TP:', tp, ' ', 'TN:', tn, ' ', 'FP:', fp, 'FN:', fn)
            print('______________________')
        
        x_test[i] = np.array(x_test[i])
        x_test[i] = x_test[i].reshape(1, 3)
        x_test[i] = x_test[i].astype('float32')
    
        predicted = model.predict(x_test[i], verbose=0)
        predicted = predicted[0][0]
        
        actual = int(y_test[i])

        if (actual == 1) and (predicted >= threshold): # True Positive
            tp += 1
        elif (actual == 0) and (predicted >= threshold): # False Positive
            fp += 1
        elif (actual == 1) and (predicted < threshold): # False Negative
            fn += 1
        elif (actual == 0) and (predicted <= threshold): # True Negative
            tn += 1

    accuracy = precision = recall = -1
    
    try:
        accuracy = float(tp + tn) / (tp + tn + fp + fn)
    except ZeroDivisionError as z:
        pass
    
    try:
        precision = float(tp) / (tp + fp)
    except ZeroDivisionError as z:
        pass

    try:
        recall = float(tp) / (tp + fn)
    except ZeroDivisionError as z:
        pass
    
    if write_to_file:
        result = str(threshold) + '\t' + str(precision) + '\t' + str(recall) + '\n'
        w = open('ffnn_train_precision_recall.dat', 'a')

        w.write('threshold\tprecision\trecall\n')
        w.write(result)
        
        w.close()
    else:
        print('Accuracy:', accuracy)
        print('Precision:', precision)
        print('Recall:', recall)

def inbuilt_metrics(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('Test precision:', score[2])
    print('Test recall:', score[3])

#Train the model
if not(custom_metrics_flag):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=3, kernel_regularizer=regularizers.l2(0.01)))#input_shape = (5, )))
    model.add(Dropout(0.5))
    model.add(Dense(32*2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.summary()

    model.compile(loss='mean_squared_error',
                  optimizer=Adam(),
                  metrics=['accuracy', precision, recall])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))

    model.save('ffnn_train_relu.h5')
    
    inbuilt_metrics(model, x_test, y_test)

else:
    model = load_model('ffnn_train_relu.h5', custom_objects={'precision':precision, 'recall':recall})
    
    custom_metrics(model, x_test, y_test, len_l, threshold, dataset_split, to_file)


sys.stdout.flush()
