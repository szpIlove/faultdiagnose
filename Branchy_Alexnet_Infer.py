#import torch
import tensorflow as tf
import keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, MaxPooling1D, Activation, Flatten, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2
from keras.callbacks import TensorBoard
import numpy as np
from config import *
from collections import OrderedDict

def infer(cORs, ep, pp):
    '''
    DNN model inference
    :param cORs: client or server
    :param pp: partition point
    :param ep: exit point
    :return: intermediate data or final result
    '''
    netPair = 'NetExit' + str(ep) + 'Part' + str(pp)
    inputs = tf.keras.Input(shape=(100,3))
    
    # load params
    LOrR = 'L' if cORs == CLIENT else 'R'
    if(ep==1 and pp==1):
        if(LOrR=='L'):
            y=tf.keras.layers.Conv1D(filters=16, kernel_size=64, strides=16, padding='same', name="conv_1")(inputs)
            y=tf.keras.layers.BatchNormalization(name="batch_1")(y)
            y=tf.keras.layers.Activation('relu',name="relu_1")(y)
            y=tf.keras.layers.MaxPooling1D(pool_size=2,name="pool_1",data_format="channels_first")(y)
            y=tf.keras.layers.Conv1D(32,kernel_size=3, strides=1, padding='same', name="conv_2")(y)
            y=tf.keras.layers.BatchNormalization(name="batch_2")(y)
            y=tf.keras.layers.Activation('relu',name="relu_2")(y)
            y=tf.keras.layers.MaxPooling1D(pool_size=2,name="pool_2", padding='valid',data_format="channels_first")(y)
            y=tf.keras.layers.Conv1D(64,kernel_size=3, strides=1, padding='same', name="conv_3")(y)
            y=tf.keras.layers.BatchNormalization(name="batch_3")(y)
            y=tf.keras.layers.Activation('relu',name="relu_3")(y)
            y=tf.keras.layers.MaxPooling1D(pool_size=2,name="pool_3", padding='valid',data_format="channels_first")(y)
            y=tf.keras.layers.Conv1D(64,kernel_size=3, strides=1, padding='same', name="conv_4")(y)
            y=tf.keras.layers.BatchNormalization(name="batch_4")(y)
            y=tf.keras.layers.Activation('relu',name="relu_4")(y)
            y=tf.keras.layers.MaxPooling1D(pool_size=2,name="pool_4", padding='valid',data_format="channels_first")(y)
            model = tf.keras.Model(inputs=inputs, outputs=y)
            return model
        if(LOrR=='R'):
            json_file = open('recv_model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            y=tf.keras.layers.Flatten(name="flat_1")(loaded_model.layers[-1].output)
            y=tf.keras.layers.Dropout(0.2,name="drop_1")(y)
            y=Dense(100,name="dense_1")(y)
            y=Activation("relu",name="relu_6")(y)
            y=Dense(units=10, activation='softmax', name="dense_2")(y)
            modelright = tf.keras.Model(inputs=loaded_model.layers[0].output, outputs=y, name='modelright')
            return modelright
    if(ep==2 and pp==1):
    
        if(LOrR=='L'):
            y=tf.keras.layers.Conv1D(filters=16, kernel_size=64, strides=16, padding='same', name="conv_1")(inputs)
            y=tf.keras.layers.BatchNormalization(name="batch_1")(y)
            y=tf.keras.layers.Activation('relu',name="relu_1")(y)
            y=tf.keras.layers.MaxPooling1D(pool_size=2,name="pool_1",data_format="channels_first")(y)
            y=tf.keras.layers.Conv1D(32,kernel_size=3, strides=1, padding='same', name="conv_2")(y)
            y=tf.keras.layers.BatchNormalization(name="batch_2")(y)
            y=tf.keras.layers.Activation('relu',name="relu_2")(y)
            y=tf.keras.layers.MaxPooling1D(pool_size=2,name="pool_2", padding='valid',data_format="channels_first")(y)
            y=tf.keras.layers.Conv1D(64,kernel_size=3, strides=1, padding='same', name="conv_3")(y)
            y=tf.keras.layers.BatchNormalization(name="batch_3")(y)
            y=tf.keras.layers.Activation('relu',name="relu_3")(y)
            y=tf.keras.layers.MaxPooling1D(pool_size=2,name="pool_3", padding='valid',data_format="channels_first")(y)
            y=tf.keras.layers.Conv1D(64,kernel_size=3, strides=1, padding='same', name="conv_4")(y)
            y=tf.keras.layers.BatchNormalization(name="batch_4")(y)
            y=tf.keras.layers.Activation('relu',name="relu_4")(y)
            y=tf.keras.layers.MaxPooling1D(pool_size=2,name="pool_4", padding='valid',data_format="channels_first")(y)
            y=tf.keras.layers.Conv1D(64,kernel_size=3, strides=1, padding='valid', name="conv_5")(y)
            y=tf.keras.layers.BatchNormalization(name="batch_5")(y)
            y=tf.keras.layers.Activation('relu',name="relu_5")(y)
            y=tf.keras.layers.MaxPooling1D(pool_size=2,name="pool_5", padding='valid',data_format="channels_first")(y)
            model = tf.keras.Model(inputs=inputs, outputs=y)
            return model
        if(LOrR=='R'):
            json_file = open('recv_model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            y=tf.keras.layers.Flatten(name="flat_1")(loaded_model.layers[-1].output)
            y=tf.keras.layers.Dropout(0.2,name="drop_1")(y)
            y=Dense(100,name="dense_1")(y)
            y=Activation("relu",name="relu_6")(y)
            y=Dense(units=10, activation='softmax', name="dense_2")(y)
            
            modelright = tf.keras.Model(inputs=loaded_model.layers[0].output, outputs=y, name='modelright')
            return modelright
            
    if(ep==3 and pp==1):
        if(LOrR=='L'):
           
            inputs = tf.keras.Input(shape=(100,3))
            y=tf.keras.layers.Conv1D(filters=16, kernel_size=64, strides=16, padding='same', name="conv_1")(inputs)
            y=tf.keras.layers.BatchNormalization(name="batch_1")(y)
            y=tf.keras.layers.Activation('relu',name="relu_1")(y)
            y=tf.keras.layers.MaxPooling1D(pool_size=2,name="pool_1",data_format="channels_first")(y)
            y=tf.keras.layers.Conv1D(32,kernel_size=3, strides=1, padding='same', name="conv_2")(y)
            y=tf.keras.layers.BatchNormalization(name="batch_2")(y)
            y=tf.keras.layers.Activation('relu',name="relu_2")(y)
            y=tf.keras.layers.MaxPooling1D(pool_size=2,name="pool_2", padding='valid',data_format="channels_first")(y)
            y=tf.keras.layers.Conv1D(64,kernel_size=3, strides=1, padding='same', name="conv_3")(y)
            y=tf.keras.layers.BatchNormalization(name="batch_3")(y)
            y=tf.keras.layers.Activation('relu',name="relu_3")(y)
            y=tf.keras.layers.MaxPooling1D(pool_size=2,name="pool_3", padding='valid',data_format="channels_first")(y)
            
            model = tf.keras.Model(inputs=inputs, outputs=y)
            return model
        if(LOrR=='R'):
            
            json_file = open('recv_model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            
            
            y=tf.keras.layers.LSTM(32,activation='tanh', recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', return_sequences=True,name="lstm_1")(loaded_model.layers[-1].output)
            y=tf.keras.layers.LSTM(100,activation='tanh', recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', return_sequences=True,name="lstm_2")(y)
            y=tf.keras.layers.LSTM(100,activation='tanh', recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', return_sequences=True,name="lstm_3")(y)
            y=tf.keras.layers.Flatten(name="flat_1")(y)
            y=tf.keras.layers.Dropout(0.2,name="drop_1")(y)
            y=Dense(100,name="dense_1")(y)
            y=Activation("relu",name="relu_6")(y)
            y=Dense(units=10, activation='softmax', name="dense_2")(y)
            
            modelright = tf.keras.Model(inputs=loaded_model.layers[0].output, outputs=y, name='modelright')
            
            return modelright

