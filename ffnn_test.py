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

custom_metrics_flag = True
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
    
model = load_model('ffnn_train_gm_chr6_mod_peak.h5', custom_objects={'precision':precision, 'recall':recall})

print('Opening file...')
f = open('gm_chr22_mod_peak.dat')
l = f.readlines()
f.close()
len_l = len(l)
index = []
for i in range(len_l):
    l[i] = l[i].split()
    #index.append([l[i][0], l[i][1]])
    del l[i][0] # delete chrStart
    del l[i][0] # delete chrEnd

for i in range(len_l):
    l[i][0] = float(l[i][0])
    l[i][1] = float(l[i][1])
    l[i][2] = float(l[i][2])
    l[i][3] = float(l[i][3])
    l[i][4] = float(l[i][4])
    #if l[i][4] == 'no\n':
    #    l[i][4] = 0.0
    #else:
    #    l[i][4] = 1.0

threshold = 0.5 # threshold for the classifier
dataset_split = 0 # give the whole file as test set

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
        x_train.append([[data[i][0], data[i][1], data[i][2], data[i][3]]])
        y_train.append(data[i][-1])
    
    for i in range(int(split * len_data), len_data):
        x_test.append([[data[i][0], data[i][1], data[i][2], data[i][3]]])
        y_test.append(data[i][-1])

    return (x_train, y_train), (x_test, y_test)

print('Loading data...')
(x_train, y_train), (x_test, y_test) = create_training_and_test_set(l, split=dataset_split)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

x_test = np.array(x_test)
y_test = np.array(y_test)

if not(custom_metrics_flag):
    x_test = x_test.reshape(len_l - int(dataset_split*len_l), 4)
    x_test = x_test.astype('float32')

y_test = y_test.astype('float32')
#y_test = keras.utils.to_categorical(y_test, 2)

def inbuilt_metrics(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('Test precision:', score[2])
    print('Test recall:', score[3])

def custom_metrics(model, index, x_test, y_test, len_l, threshold, dataset_split, write_to_file=False):
    tp = 0 # True Pos
    fp = 0 # False Pos
    tn = 0 # True Neg
    fn = 0 # False Neg
    
    # write predictions to a file?
    if write_to_file:
        result = ''
    for i in xrange(len(x_test)):
        if not(i%100000):
            print(str(100 * float(i) / (len_l - dataset_split*len_l)) + '%')
            print('TP:', tp, 'TN:', tn, 'FP:', fp, 'FN:', fn)
            print('______________________')
        
        x_t = np.array(x_test[i])
        x_t = x_t.reshape(1, 4)
        x_t = x_t.astype('float32')
        
        if ((x_t[0][1] == 0.0) and (x_t[0][2] == 0.0) and (x_t[0][3] == 0.0)):
            predicted = 0
        else:
            predicted = model.predict(x_t, verbose=0)
            predicted = predicted[0][0]
            #if predicted[0][0] > predicted[0][1]:
            #    predicted = predicted[0][0]
            #else:
            #    predicted = predicted[0][1]
        
        #actual = 1
        #if y_test[i][0] > y_test[i][1]:
        #    actual = 0
        actual = int(y_test[i])
        #print x_test[i], actual, predicted
        if (actual == 1) and (predicted >= threshold): # True Positive
            tp += 1
        elif (actual == 0) and (predicted >= threshold): # False Positive
            fp += 1
        elif (actual == 1) and (predicted < threshold): # False Negative
            fn += 1
        elif (actual == 0) and (predicted <= threshold): # True Negative
            tn += 1
        #'''
        if write_to_file:
            result += str(x_t) + '\t' + str(actual) + '\t' + str(predicted) + '\n'
            #result += index[i][0] + '\t' + index[i][1] + '\t' + str(x_test[i]) + '\t' + str(y_test[i]) + '\t' + str(actual) + '\t' + str(predicted) + '\n'
            #result += index[i][0] + '\t' + index[i][1] + '\t' + str(actual) + '\t' + str(predicted) + '\n'

    if write_to_file:
        w = open('ffnn_test_output_gm_chr22.dat', 'w')
        
        w.write('chrStart\tchrEnd\tactual\tpredicted\n')
        w.write(result)
        
        w.close()

    print threshold
    print('TP:', tp, ' ', 'TN:', tn, ' ', 'FP:', fp, 'FN:', fn)
    
    precision = recall = -1
    
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
    
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('_________')

if custom_metrics_flag:
    for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print th
        custom_metrics(model, index, x_test, y_test, len_l, th, dataset_split, to_file)
else:
    inbuilt_metrics(model, x_test, y_test)
