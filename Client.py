import tensorflow as tf
import numpy as np
import math
import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


# 自定义的数据集类

from Dataset import Dataset

class Clients:
    def __init__(self, input_shape, num_classes, learning_rate, clients_num):
        self.graph = tf.Graph()
        self.sess = tf.compat.v1.Session(graph=self.graph)
        
        with self.graph.as_default():
            '''model = Sequential()
            model.add(LSTM(30, input_shape=(100,3)))
            model.add(Dropout(0.2))
            model.add(Dense(7, activation='softmax'))
            model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy',  metrics=['accuracy'])'''
            inputs = tf.keras.Input(shape=(100, 3))
            y = tf.keras.layers.LSTM(30, name="lstm_1")(inputs)
            y = tf.keras.layers.Dropout(0.2, name="drop_1")(y)
            y = Dense(units=7, activation='softmax', name="dense_1")(y)
            model = tf.keras.Model(inputs=inputs, outputs=y, name='model')
            model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy',
                          metrics=['accuracy'])
                                    
        self.model = model
        
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

        # Load Cifar-10 dataset
        # NOTE: len(self.dataset.train) == clients_num
        # 加载数据集。对于训练集：`self.dataset.train[56]`可以获取56号client的数据集
        # `self.dataset.train[56].next_batch(32)`可以获取56号client的一个batch，大小为32
        # 对于测试集，所有client共用一个测试集，因此：
        # `self.dataset.test.next_batch(1000)`将获取大小为1000的数据集（无随机）
        self.dataset = Dataset(split=clients_num)
        print("dataset.train-%d, clients_num-%d" % (len(self.dataset.train), clients_num))

    """
        Predict the testing set, and report the acc and loss
        预测测试集，返回准确率和loss

        num: number of testing instances
    """
    def run_test(self, num):
        with self.graph.as_default():
            batch_x, batch_y = self.dataset.test.next_batch(num)
            score = self.model.evaluate(batch_x, batch_y)
        return score

    def run_predict(self, num, layermoder):
        with self.graph.as_default():
            batch_x, batch_y = self.dataset.test.next_batch(num)
            layerout = layermoder.predict(batch_x)
        return layerout

    def run_confusion_matrix(self, num):
        with self.graph.as_default():
            batch_x, batch_y = self.dataset.test.next_batch(num)
            predict_classes = np.argmax(self.model.predict(batch_x), axis=1)
            true_classes = np.argmax(batch_y, 1)
            plot_url = confusion_matrix.plot_confusion_matrix_fed(true_classes, predict_classes, save_flg=True)
        return plot_url

    def train_epoch(self, cid, batch_size=64):
        """
            Train one client with its own data for one epoch
            用`cid`号的client的数据对模型进行训练
            cid: Client id
        """
        clents = {0:5000,1:4000,2:3000,3:2020,4:4020,5:3000,6:2000,7:2500,8:2086,9:2800}
        dataset = self.dataset.train[cid]
        # datasetval = self.dataset.valid[cid]
        checkpoint = ModelCheckpoint('federatedlearning_10.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        with self.graph.as_default():
            train_x, train_y = dataset.next_batch(clents[cid])
            history = self.model.fit(train_x, train_y, validation_split=0.1, epochs=1,
                                    batch_size=batch_size, verbose=1
                                    # , callbacks=callbacks_list
                                    )
        return history

    # 返回计算图中所有可训练的变量值
    def get_client_vars(self):
        """ Return all of the variables list """
        with self.graph.as_default():
            client_vars = self.sess.run(tf.trainable_variables())
        return client_vars

    def set_global_vars(self, global_vars):
        """ Assign all of the variables with global vars """
        with self.graph.as_default():
            '''把整个模型中的变量值都加载，可以用tf.trainable_variables()
            获取计算图中的所有可训练变量（一个list），保证它和global_vars的顺序对应后'''
            all_vars = tf.trainable_variables()
            for variable, value in zip(all_vars, global_vars):
                ''' 把一个Tensor的值加载到模型变量中
                    variable.load(tensor, sess)
                    将tensor（类型为tf.Tensor）的值赋值给variable（类型为tf.Varibale），sess是tf.Session
                '''
                variable.load(value, self.sess)
                #tf.assign(variable, value)

    def choose_clients(self, ratio=1.0):
        """ randomly choose some clients 
            随机选择`ratio`比例的clients，返回编号（也就是下标）
        """
        client_num = self.get_clients_num()
        choose_num = math.floor(client_num * ratio)
        # 对0-5之间的序列进行随机排序
        return np.random.permutation(client_num)[:choose_num]

    def get_clients_num(self):
        """ 返回clients的数量 
            len(self.dataset.train) == clients_num
        """        
        return len(self.dataset.train)
    