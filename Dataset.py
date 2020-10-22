import os
import numpy as np
import pandas as pd
import scipy.io as sio

from sklearn.model_selection import train_test_split

class BatchGenerator:
    def __init__(self, x, yy):
        self.x = x
        self.y = yy
        self.size = len(x)
        self.random_order = list(range(len(x)))
        # np.random.shuffle(x)：在原数组上进行，改变自身序列，无返回值。
        # np.random.permutation(x)：不在原数组上进行，返回新的数组，不改变自身数组。
        np.random.shuffle(self.random_order)
        self.start = 0
        return

    def next_batch(self, batch_size):
        # 当剩余数据不够一个batch时，从开始的位置补齐
        if self.start + batch_size >= len(self.random_order):
            overflow = (self.start + batch_size) - len(self.random_order)
            perm0 = self.random_order[self.start:] +\
                 self.random_order[:overflow]
            self.start = overflow
        else:
            perm0 = self.random_order[self.start:self.start + batch_size]
            self.start += batch_size

        # assert len(perm0) == batch_size

        return self.x[perm0], self.y[perm0]

    # support slice
    def __getitem__(self, val):
        return self.x[val], self.y[val]


class Dataset(object):
    def __init__(self, split=0):
        
        TIMESERIES_LENGTH = 100
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        target_path = os.path.join(BASE_DIR, "files/")
        f = open(target_path+"12kdata.txt","r", encoding='UTF-8')
        lines = f.readlines()
        datas = {}
        fault_type = 'undefined'
        description = ''
        # fault_types = ['DE_B', 'DE_IR', 'DE_OR', 'FE_B', 'FE_IR', 'FE_B']
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
                continue
            filename, suffix = line.split('.')
            print('Loading data {0} {1} {2}'.format(filename,fault_type,description))
            params = filename.split('_')
            data_no = params[-1]
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            target_path = os.path.join(BASE_DIR, "files", "CaseWesternReserveUniversityData/")
            mat_data = sio.loadmat(target_path +filename)
            if data_no == '097' or data_no == '098' or data_no == '099' or data_no == '100':
                features_considered = map(lambda key_str:key_str.format(data_no), ["X{0}_DE_time", "X{0}_DE_time", "X{0}_DE_time"])
            else:
                features_considered = map(lambda key_str:key_str.format(data_no), ["X{0}_DE_time", "X{0}_FE_time", "X{0}_BA_time"])
            current_features = np.array([mat_data[feature].flatten() for feature in features_considered])
            features = np.concatenate((features, current_features), axis=1)
            # multidimensional_timeseries = np.hstack(current_features)
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
            fault_type,description, length = row['fault_type'],row['description'], row['length']
            current_features = features[:,start_index:start_index+length]
            multidimensional_timeseries = current_features.T
            start_index += length
            data = [multidimensional_timeseries[i:i+TIMESERIES_LENGTH] for i in range(0, length - TIMESERIES_LENGTH, 100)]
            if fault_type not in datas:
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
        datas['NORMAL']['X'] = choice(datas['NORMAL']['X'], 4845)

        label_placeholder = np.zeros(7, dtype=int)
        x_data = np.empty(shape=(0,TIMESERIES_LENGTH,3))
        y_data = np.empty(shape=(0,7),dtype=int)
        DATASET_SIZE = 0
        BATCH_SIZE = 16
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
        
        '''
        # 下采样
        X_data_p = np.array(x_data)
        y_data_p = np.array(y_data)
        nsamples, nx, ny = X_data_p.shape
        d2_train_dataset = X_data_p.reshape(nsamples,nx*ny)
        d2_label_dataset=np.argmax(y_data_p,axis=1)
        #X_tsne = tsne.fit_transform(d2_train_dataset)

        nm1 = NearMiss(version=1)
        X_resampled_nm1, y_resampled = nm1.fit_sample(d2_train_dataset, d2_label_dataset)
        m = sorted(Counter(y_resampled).items())
        nsamples2, nx2 = X_resampled_nm1.shape
        d3_train_dataset = X_resampled_nm1.reshape(nsamples2,nx,ny)
        d3_label_dataset = np.empty(shape=(0,7),dtype=int)
        for ki,vi in m:    
            label = np.copy(label_placeholder)
            label[ki] = 1 # one-hot encode
            labels = np.repeat([label], vi, axis=0)
            d3_label_dataset = np.concatenate((d3_label_dataset, labels))
        print(d3_train_dataset.shape)
        print(d3_label_dataset.shape)
        '''
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1)        
        print("Dataset: train-%d,test-%d" % (len(x_train),len(x_test)))

        if split == 0:
            self.train = BatchGenerator(x_train, y_train)
            # self.valid = BatchGenerator(valid_X, valid_y)
        else:
            self.train = self.splited_batch(x_train, y_train, split)
            # self.valid = self.splited_batch(valid_X, valid_y, split)

        self.test = BatchGenerator(x_test, y_test)

    def splited_batch(self, x_data, y_data, count):
        res = []
        clents = [5000,4000,3000,2020,4020,3000,2000,2500,2086,2800]
        l = len(x_data)
        k,j = 0,0
        for i in range(count):
            res.append(
                BatchGenerator(x_data[k:k+clents[j]],
                               y_data[k:k+clents[j]]))
            k=k+clents[j]
            j=j+1
            
        return res
