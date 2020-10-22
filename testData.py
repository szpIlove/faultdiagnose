from tensorflow.keras.models import Sequential
#from keras import models
#from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM, CuDNNLSTM, CuDNNGRU, RNN
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM, RNN, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.model_selection import train_test_split


import os
import gc
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import numpy as np
 
import numpy as np
import pandas as pd
import scipy.io as sio
import pickle

TIMESERIES_LENGTH = 100

f = open("edgedata_0_1_2.txt","r",encoding='UTF-8')
lines = f.readlines()
datas = {}
fault_type = 'undefined'
description = ''
fault_typeList = []
descriptionList = []
statistics_df = pd.DataFrame(columns=['filename','fault_type', 'description','length','DE_min','DE_max','DE_mean','DE_std','FE_min','FE_max','FE_mean','FE_std','BA_min','BA_max','BA_mean','BA_std'])
features = np.empty(shape=(3,0))
def npstatistics(data):
    return [data.min(), data.max(), data.mean(), data.std()]
for line in lines:
    line = line.strip()
    if len(line) == 0 or line.startswith('#'):
        continue
    if line.startswith('faultType'):
        comments = line.split(' ')
        fault_type = comments[1]
        description = comments[2]
        print("-------------------")
        print(description)
        descriptionList.append(description)
        continue
    filename, suffix = line.split('.')
    print('Loading data {0} {1} {2}'.format(filename,fault_type,description))
    params = filename.split('_')
    data_no = params[-1]
    print(data_no)
    #mat_data = sio.loadmat('./CaseWesternReserveUniversityData/'+filename)
    mat_data = sio.loadmat('./CaseWesternReserveUniversityData/'+filename)
    features_considered = map(lambda key_str:key_str.format(data_no), ["X{0}_DE_time", "X{0}_FE_time", "X{0}_DE_time"])
    current_features = np.array([mat_data[feature].flatten() for feature in features_considered])
    features = np.concatenate((features, current_features), axis=1)  #列拼接
    
    data_size = len(mat_data["X{0}_DE_time".format(data_no)]) # current file timeseries length
    statistics = [filename, fault_type, description, data_size]
    statistics += npstatistics(current_features[0])
    statistics += npstatistics(current_features[1])
    statistics += npstatistics(current_features[2])
    
    statistics_df.loc[statistics_df.size] = statistics

f.close()
print("\nStatistics:")
print(statistics_df.head())

def normalize(data):
    mean = data.mean()
    std = data.std()
    return (data-mean)/std
features[0] = normalize(features[0])
features[1] = normalize(features[1])
features[2] = normalize(features[2])

start_index = 0
for index, row in statistics_df.iterrows():
    fault_type, length = row['fault_type'], row['length']
    current_features = features[:,start_index:start_index+length]
    multidimensional_timeseries = current_features.T
    start_index += length
    data = [multidimensional_timeseries[i:i+TIMESERIES_LENGTH] for i in range(0, length - TIMESERIES_LENGTH, 100)]
    if fault_type not in datas:
        fault_typeList.append(fault_type)
        description=descriptionList[fault_typeList.index(fault_type)]
        datas[fault_type] = {
            'fault_type':fault_type,
            'description':description,
            'X':np.empty(shape=(0, TIMESERIES_LENGTH, 3))
        }
        
    datas[fault_type]['X'] = np.concatenate((datas[fault_type]['X'], data))

# %%
# random choice
def choice(dataset, size):
    return dataset[np.random.choice(dataset.shape[0], size, replace=False), :]
# make data balance
#datas['DE_OR']['X'] = choice(datas['DE_OR']['X'], 14500)
#datas['FE_OR']['X'] = choice(datas['FE_OR']['X'], 14500)

cloud_num=len(fault_typeList)
label_placeholder = np.zeros(cloud_num, dtype=int)
x_data = np.empty(shape=(0,TIMESERIES_LENGTH,3),dtype=np.float32)

y_data = np.empty(shape=(0,cloud_num),dtype=np.int64)

DATASET_SIZE = 0
#BATCH_SIZE = 16
BATCH_SIZE = 32
BUFFER_SIZE = 10000
for index, (key, value) in enumerate(datas.items()):
    sample_size = len(value['X'])
    DATASET_SIZE += sample_size
    print("{0} {1} {2}".format(value['fault_type'], value['description'], sample_size))
    label = np.copy(label_placeholder)
    label[index] = 1 # one-hot encode
    
    x_data = np.concatenate((x_data, value['X']))
    labels = np.repeat([label], sample_size, axis=0)
    y_data = np.concatenate((y_data, labels))
total_data = [(x_data[i], y_data[i]) for i in range(0,len(x_data))]
np.random.shuffle(total_data)
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
def testData():
    return X_test
def labelData():
    return y_test