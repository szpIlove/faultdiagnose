from config import *

branch1 = ['conv_1', 'norm_1','relu_1', 'pool_1', 'conv_2', 'norm_2','relu_2', 'pool_2', 'conv_3', 'norm_3','relu_3', 'pool_3', 'conv_4', 'norm_4','relu_4', 'pool_4',  'drop_1', 'dense_1', 'relu_6', 'dense_2']
branch2 = ['conv_1', 'norm_1','relu_1', 'pool_1', 'conv_2', 'norm_2','relu_2', 'pool_2', 'conv_3', 'norm_3','relu_3', 'pool_3', 'conv_4', 'norm_4','relu_4', 'pool_4', 'conv_5', 'norm_5','relu_5', 'pool_5' , 'drop_1', 'dense_1', 'relu_6', 'dense_2']
branch3 = ['conv_1', 'norm_1','relu_1', 'pool_1', 'conv_2', 'norm_2','relu_2', 'pool_2', 'conv_3', 'norm_3','relu_3', 'pool_3', 'conv_4', 'norm_4','relu_4', 'pool_4', 'conv_5', 'norm_5','relu_5', 'pool_5','lstm_1','lstm_2','lstm_3','drop_1', 'dense_1', 'relu_6', 'dense_2']
	   
branch1_partition_index = [15]
branch2_partition_index = [19]
branch3_partition_index = [19]

partition_point_number = [1, 1, 1]

branches_info = [(branch1, branch1_partition_index), (branch2, branch2_partition_index),
                 (branch3, branch3_partition_index)]

# Bytes
model_size = {
    
    'branch1_part1L': 120087,
    'branch1_part1R': 51804,
    'branch1_part2L': 131559,
    'branch1_part2R': 63325,
    'branch2_part1L': 1249966,
    'branch2_part1R': 227393,
    'branch2_part2L': 1249966,
    'branch2_part2R': 227393,
    'branch2_part3L': 1471614,
    'branch2_part3R': 5723,
    'branch3_part1L': 43870657,
    'branch3_part1R': 8870657,
    'branch3_part2L': 1249896,
    'branch3_part2R': 92640835,
    'branch3_part3L': 9806701,
    'branch3_part3R': 84083973
}


###############################################
# Mobile device side time prediction class
###############################################
class DeviceTime:
    def __init__(self):
        self.branch1 = {
            'conv_1': self.device_conv(3, (5 * 5 * 3) ** 2 * 64),
            'norm_1': self.device_lrn(64 * 15 * 15),
            'relu_1': self.device_relu(63 * 32 * 32),
            'pool_1': self.device_pool(64 * 32 * 32, 64 * 15 * 15),
            
            'conv_2':self.device_conv(64, (5 * 5 * 64) ** 2 * 192),
            'relu_2': self.device_relu(192 * 13 * 13),
            'norm_2': self.device_lrn(192 * 13 * 13),
            'pool_2': self.device_pool(192 * 13 * 13, 192 * 6 * 6),
            
            'conv_3': self.device_conv(192, (3 * 3 * 192) ** 2 * 384),
            'relu_3': self.device_relu(192 * 13 * 13),
            'norm_3': self.device_relu(384 * 6 * 6),
            'pool_3': self.device_pool(192 * 13 * 13, 192 * 6 * 6),
			
            'conv_4': self.device_conv(384, (3 * 3 * 384) ** 2 * 256),
            'relu_4': self.device_relu(256 * 6 * 6),
            'norm_4': self.device_relu(384 * 6 * 6),
            'pool_4': self.device_pool(192 * 13 * 13, 192 * 6 * 6),
            
            'drop_1':self.device_dropout(1024),
            'dense_1':self.device_linear(1568, 32),
            'relu_6': self.device_relu(63 * 32 * 32),
            'dense_2':self.device_linear(1568, 10),
        }
        self.branch2 = {
            
            'conv_1': self.device_conv(3, (5 * 5 * 3) ** 2 * 64),
            'norm_1': self.device_lrn(64 * 15 * 15),
            'relu_1': self.device_relu(63 * 32 * 32),
            'pool_1': self.device_pool(64 * 32 * 32, 64 * 15 * 15),
            
            'conv_2':self.device_conv(64, (5 * 5 * 64) ** 2 * 192),
            'relu_2': self.device_relu(192 * 13 * 13),
            'norm_2': self.device_lrn(192 * 13 * 13),
            'pool_2': self.device_pool(192 * 13 * 13, 192 * 6 * 6),
            
            'conv_3': self.device_conv(192, (3 * 3 * 192) ** 2 * 384),
            'relu_3': self.device_relu(192 * 13 * 13),
            'norm_3': self.device_relu(384 * 6 * 6),
            'pool_3': self.device_pool(192 * 13 * 13, 192 * 6 * 6),
			
            'conv_4': self.device_conv(384, (3 * 3 * 384) ** 2 * 256),
            'relu_4': self.device_relu(256 * 6 * 6),
            'norm_4': self.device_relu(384 * 6 * 6),
            'pool_4': self.device_pool(192 * 13 * 13, 192 * 6 * 6),
            
            'conv_5': self.device_conv(256, (3 * 3 * 256) ** 2 * 256),
            'relu_5': self.device_relu(256 * 6 * 6),
            'norm_5': self.device_relu(384 * 6 * 6),
            'pool_5': self.device_pool(256 * 6 * 6, 256 * 2 * 2),
            #'flat_1': 
            'drop_1':self.device_dropout(1024),
            'dense_1':self.device_linear(1568, 32),
            'relu_6': self.device_relu(63 * 32 * 32),
            'dense_2':self.device_linear(1568, 10),
        }
        self.branch3 = {
            'conv_1': self.device_conv(3, (5 * 5 * 3) ** 2 * 64),
            'norm_1': self.device_lrn(64 * 15 * 15),
            'relu_1': self.device_relu(63 * 32 * 32),
            'pool_1': self.device_pool(64 * 32 * 32, 64 * 15 * 15),
            
            'conv_2':self.device_conv(64, (5 * 5 * 64) ** 2 * 192),
            'relu_2': self.device_relu(192 * 13 * 13),
            'norm_2': self.device_lrn(192 * 13 * 13),
            'pool_2': self.device_pool(192 * 13 * 13, 192 * 6 * 6),
            
            'conv_3': self.device_conv(192, (3 * 3 * 192) ** 2 * 384),
            'relu_3': self.device_relu(192 * 13 * 13),
            'norm_3': self.device_relu(384 * 6 * 6),
            'pool_3': self.device_pool(192 * 13 * 13, 192 * 6 * 6),
			
            'conv_4': self.device_conv(384, (3 * 3 * 384) ** 2 * 256),
            'relu_4': self.device_relu(256 * 6 * 6),
            'norm_4': self.device_relu(384 * 6 * 6),
            'pool_4': self.device_pool(192 * 13 * 13, 192 * 6 * 6),
            
            'conv_5': self.device_conv(256, (3 * 3 * 256) ** 2 * 256),
            'relu_5': self.device_relu(256 * 6 * 6),
            'norm_5': self.device_relu(384 * 6 * 6),
            'pool_5': self.device_pool(256 * 6 * 6, 256 * 2 * 2),
            'lstm_1': self.device_lstm(192 * 13 * 13, 192 * 6 * 6),
            'lstm_2': self.device_lstm(192 * 100 * 100, 192 * 6 * 6),
            'lstm_3': self.device_lstm(192 * 100 * 100, 192 * 6 * 6),
            'drop_1':self.device_dropout(1024),
            'dense_1':self.device_linear(1568, 32),
            'relu_6': self.device_relu(63 * 32 * 32),
            'dense_2':self.device_linear(1568, 10),
        }
        self.branches = [self.branch1, self.branch2, self.branch3]

    # time predict function
    def device_lrn(self, data_size):
        return 9.013826444839453e-08 * data_size + 0.0013616842338199375

    def device_pool(self, input_data_size, output_data_size):
        return 1.1864462944013584e-08 * input_data_size - 2.031421398089179e-09 * output_data_size + 0.0001234705954153948

    def device_relu(self, input_data_size):
        return 6.977440389615429e-09 * input_data_size + 0.0005612587990019447
    
    def device_lstm(self,input_data_size, output_data_size):
        return 1.1864462944013584e-08 * input_data_size - 2.031421398089179e-09 * output_data_size + 2.6501234705954153948

    def device_dropout(self, input_data_size):
        return 9.341929545685408e-08 * input_data_size + 0.0007706006740869353

    def device_linear(self, input_data_size, output_data_size):
        return 1.1681471979101294e-08 * input_data_size + 0.00029824333961563884 * output_data_size - 0.0011913997548602204

    def device_conv(self, feature_map_amount, compution_each_pixel):
        return 0.00020423363723714956 * feature_map_amount + 4.2077298118910815e-11 * compution_each_pixel + 0.055591776113868925

    def device_model_load(self, model_size):
        return 12.558441818370891e-09 * model_size + 0.401395207253916772

    # tool
    def predict_time(self, branch_number, partition_point_number):
        '''
        :param branch_number: the index of branch
        :param partition_point_number: the index of partition point
        :return:
        '''
        branch_layer, partition_point_index_set = branches_info[branch_number]
        partition_point = partition_point_index_set[partition_point_number]
        layers = branch_layer[:partition_point + 1]
        time_dict = self.branches[branch_number]

        time = 0
        for layer in layers:
            time += time_dict[layer]
        return time


###############################################
# Edge server side time prediction class
###############################################
class ServerTime:
    def __init__(self):
        self.branch1 = {
            
            'conv_1': self.server_conv(3, (5 * 5 * 3) ** 2 * 64),
            'norm_1': self.server_lrn(64 * 15 * 15),
            'relu_1': self.server_relu(63 * 32 * 32),
            'pool_1': self.server_pool(64 * 32 * 32, 64 * 15 * 15),
            
            'conv_2': self.server_conv(64, (5 * 5 * 64) ** 2 * 192),
            'relu_2': self.server_relu(192 * 13 * 13),
            'norm_2': self.server_lrn(192 * 13 * 13),
            'pool_2': self.server_pool(192 * 13 * 13, 192 * 6 * 6),
            
            'conv_3': self.server_conv(192, (3 * 3 * 192) ** 2 * 384),
            'relu_3': self.server_relu(192 * 13 * 13),
            'norm_3': self.server_relu(384 * 6 * 6),
            'pool_3': self.server_pool(192 * 13 * 13, 192 * 6 * 6),
			
            'conv_4': self.server_conv(384, (3 * 3 * 384) ** 2 * 256),
            'relu_4': self.server_relu(256 * 6 * 6),
            'norm_4': self.server_relu(384 * 6 * 6),
            'pool_4': self.server_pool(192 * 13 * 13, 192 * 6 * 6),
            
            'drop_1': self.server_dropout(1024),
            'dense_1':self.server_linear(1568, 32),
            'relu_6': self.server_relu(63 * 32 * 32),
            'dense_2':self.server_linear(1568, 10),
        }
        self.branch2 = {
            'conv_1': self.server_conv(3, (5 * 5 * 3) ** 2 * 64),
            'norm_1': self.server_lrn(64 * 15 * 15),
            'relu_1': self.server_relu(63 * 32 * 32),
            'pool_1': self.server_pool(64 * 32 * 32, 64 * 15 * 15),
            
            'conv_2':self.server_conv(64, (5 * 5 * 64) ** 2 * 192),
            'relu_2': self.server_relu(192 * 13 * 13),
            'norm_2': self.server_lrn(192 * 13 * 13),
            'pool_2': self.server_pool(192 * 13 * 13, 192 * 6 * 6),
            
            'conv_3': self.server_conv(192, (3 * 3 * 192) ** 2 * 384),
            'relu_3': self.server_relu(192 * 13 * 13),
            'norm_3': self.server_relu(384 * 6 * 6),
            'pool_3': self.server_pool(192 * 13 * 13, 192 * 6 * 6),
			
            'conv_4': self.server_conv(384, (3 * 3 * 384) ** 2 * 256),
            'relu_4': self.server_relu(256 * 6 * 6),
            'norm_4': self.server_relu(384 * 6 * 6),
            'pool_4': self.server_pool(192 * 13 * 13, 192 * 6 * 6),
            
            'conv_5': self.server_conv(256, (3 * 3 * 256) ** 2 * 256),
            'relu_5': self.server_relu(256 * 6 * 6),
            'norm_5': self.server_relu(384 * 6 * 6),
            'pool_5': self.server_pool(256 * 6 * 6, 256 * 2 * 2),
            #'flat_1': 
            'drop_1':self.server_dropout(1024),
            'dense_1':self.server_linear(1568, 32),
            'relu_6': self.server_relu(63 * 32 * 32),
            'dense_2':self.server_linear(1568, 10),
        }
        self.branch3 = {
            
            'conv_1': self.server_conv(3, (5 * 5 * 3) ** 2 * 64),
            'norm_1': self.server_lrn(64 * 15 * 15),
            'relu_1': self.server_relu(63 * 32 * 32),
            'pool_1': self.server_pool(64 * 32 * 32, 64 * 15 * 15),
            
            'conv_2':self.server_conv(64, (5 * 5 * 64) ** 2 * 192),
            'relu_2': self.server_relu(192 * 13 * 13),
            'norm_2': self.server_lrn(192 * 13 * 13),
            'pool_2': self.server_pool(192 * 13 * 13, 192 * 6 * 6),
            
            'conv_3': self.server_conv(192, (3 * 3 * 192) ** 2 * 384),
            'relu_3': self.server_relu(192 * 13 * 13),
            'norm_3': self.server_relu(384 * 6 * 6),
            'pool_3': self.server_pool(192 * 13 * 13, 192 * 6 * 6),
			
            'conv_4': self.server_conv(384, (3 * 3 * 384) ** 2 * 256),
            'relu_4': self.server_relu(256 * 6 * 6),
            'norm_4': self.server_relu(384 * 6 * 6),
            'pool_4': self.server_pool(192 * 13 * 13, 192 * 6 * 6),
            
            'conv_5': self.server_conv(256, (3 * 3 * 256) ** 2 * 256),
            'relu_5': self.server_relu(256 * 6 * 6),
            'norm_5': self.server_relu(384 * 6 * 6),
            'pool_5': self.server_pool(256 * 6 * 6, 256 * 2 * 2),
            #'flat_1': 
            'lstm_1': self.server_lstm(192 * 13 * 13, 192 * 6 * 6),
            'lstm_2': self.server_lstm(192 * 100 * 100, 192 * 6 * 6),
            'lstm_3': self.server_lstm(192 * 100 * 100, 192 * 6 * 6),
            'drop_1':self.server_dropout(1024),
            'dense_1':self.server_linear(1568, 32),
            'relu_6': self.server_relu(63 * 32 * 32),
            'dense_2':self.server_linear(1568, 10),
        }
		
        self.branches = [self.branch1, self.branch2, self.branch3]

    def server_lrn(self, data_size):
        return 2.111544033139625e-08 * data_size + 0.0285872721707483

    def server_pool(self, input_data_size, output_data_size):
        return -3.08201145e-10 * input_data_size + 1.19458883e-09 * output_data_size - 0.0010152380964514613

    def server_lstm(self, input_data_size, output_data_size):
        return -3.08201145e-10 * input_data_size + 4.39458883e-09 * output_data_size + 0.7400152380964514613

    def server_relu(self, input_data_size):
        return 2.332339368254984e-09 * input_data_size + 0.005070494191853819

    def server_dropout(self, input_data_size):
        return 3.962833398808942e-09 * input_data_size + 0.015458175165054516

    def server_linear(self, input_data_size, output_data_size):
        return 9.843676646891836e-12 * input_data_size + 4.0100716666407315e-07 * output_data_size + 0.015619779485748695

    def server_conv(self, feature_map_amount, compution_each_pixel):
        return 1.513486447521604e-06 * feature_map_amount + 4.4890001480985655e-12 * compution_each_pixel + 0.009816023641653768

    def server_model_load(self, model_size):
        return 14.753178793348365e-10 * model_size + 0.600678369983568624

    # tool
    def predict_time(self, branch_number, partition_point_number):
        '''
        :param branch_number: the index of branch
        :param partition_point_number: the index of partition point
        :return:
        '''
        branch_layer, partition_point_index_set = branches_info[branch_number]
        partition_point = partition_point_index_set[partition_point_number]
        layers = branch_layer[partition_point + 1:]
        time_dict = self.branches[branch_number]

        time = 0
        for layer in layers:
            time += time_dict[layer]
        return time


class OutputSizeofPartitionLayer:
    # float32 which is 4B(32 bits)
    branch1 = {
        'pool_4': 64 * 15 * 15 * 32,
        'pool1': 32 * 7 * 7 * 32,
    }
    branch2 = {
        'pool_5': 64 * 15 * 15 * 32,
        'pool1': 192 * 6 * 6 * 32,
        'pool2': 32 * 2 * 2 * 32,
    }
    branch3 = {
        'pool_5': 64 * 15 * 15 * 32,
        'pool1': 192 * 6 * 6 * 32,
        'pool2': 256 * 2 * 2 * 32,
    }
    branches = [branch1, branch2, branch3]

    @classmethod
    def output_size(cls, branch_number, partition_point_number):
        '''
        :return:unit(bit)
        '''
        branch_layer, partition_point_index_set = branches_info[branch_number]
        partition_point = partition_point_index_set[partition_point_number]
        layer = branch_layer[partition_point]
        outputsize_dict = cls.branches[branch_number]
        return outputsize_dict[layer]
