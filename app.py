from __future__ import absolute_import, division, print_function, unicode_literals

import base64
import io
import os
import time
import zipfile

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import tensorflow as tf
import thriftpy2 as thriftpy
from flask import Flask, render_template, request
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Activation
from thriftpy2.rpc import make_client
from thriftpy2.rpc import make_server

import confusion_matrix
import faultType_data_fed_plot
import faultType_data_plot
from Branchy_Alexnet_Infer import infer
import math
from Client import Clients
from tqdm import tqdm
from Optimize import Optimize
from config import *
from testData import testData, labelData

app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def npstatistics(data):
  return [data.min(), data.max(), data.mean(), data.std()]

def normalize(data):
  mean = data.mean()
  std = data.std()
  return (data - mean) / std

def choice(dataset, size):
  return dataset[np.random.choice(dataset.shape[0], size, replace=False), :]

@app.route("/", methods=['POST', "GET"])
def index():
    return render_template('base1.html')

@app.route("/index")
def index2():
    return render_template('index.html')

@app.route("/dataprocess")
def dataprocess():
  target_path = os.path.join(BASE_DIR, "files", "CaseWesternReserveUniversityData")
  filelist = os.listdir(target_path)
  descriptlist = []
  for faultType in filelist:
    faultobj = []
    fault_url = faultType.split('.')
    fault_name = fault_url[0].split('_');
    if len(fault_name) > 2:
      faultDescript = fault_name[0]
      if fault_name[1] == 'Drive':
        faultDescript += '驱动端'
      elif fault_name[1] == 'Fan':
        faultDescript += '风扇端'
      if fault_name[3].startswith('B'):
        faultDescript += '滚动体'
      elif fault_name[3].startswith('I'):
        faultDescript += '内圈'
      elif fault_name[3].startswith('O'):
        faultDescript += '外圈'
      faultDescript += '故障'
    else:
      faultDescript = '正常数据'
    faultobj.append(faultType)
    faultobj.append(faultDescript)
    descriptlist.append(faultobj)
  return render_template('dataprocessfed.html', descriptlist=descriptlist)

@app.route("/train/federated")
def trainfed():
    return render_template('trainfed.html')

@app.route("/display/federated")
def displayfed():
    return render_template('distributefed.html')

@app.route("/display/startfed")
def startdisplayfed():
    target_path = os.path.join(BASE_DIR, "files/")
    with open(target_path+"layer_fed_url.txt", "r") as f:
      plot_disp_fed_url = f.readline()
    with open(target_path+"history_fed.txt", "r") as f:
      history_fed = f.readline()
    history = eval(history_fed)
    return render_template('modeldisplayfed.html', plot_disp_fed_url=plot_disp_fed_url,testAcc=history['acc'][10])

@app.route("/train/curvefed")
def traincurvefed():
  target_path = os.path.join(BASE_DIR, "files/")
  with open(target_path+"history_fed.txt", "r") as f:
    history_fed = f.readline()
  history = eval(history_fed)
  def plot_accuracy(history):
    # plt.plot(history.history['acc'])
    plt.plot(history['acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Test'], loc='upper left')
    img7 = io.BytesIO()
    plt.savefig(img7, format='png')
    img7.seek(0)
    plt.close()
    train_fed_acc_url = base64.b64encode(img7.getvalue()).decode()
    return train_fed_acc_url

  def plot_loss(history):
    # plt.plot(history.history['loss'])
    plt.plot(history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Test'], loc='upper right')
    img8 = io.BytesIO()
    plt.savefig(img8, format='png')
    img8.seek(0)
    plt.close()
    train_fed_loss_url = base64.b64encode(img8.getvalue()).decode()
    return train_fed_loss_url

  train_fed_loss_url = plot_loss(history)
  train_fed_acc_url = plot_accuracy(history)
  return render_template('modelTrainfed.html', train_fed_acc_url=train_fed_acc_url,
                         train_fed_loss_url=train_fed_loss_url, testAcc=history['acc'][10])

@app.route("/train/startfed")
def startfed():
  def buildClients(num):
    learning_rate = 0.001
    num_input = 500  # image shape: 32*32
    num_input_channel = 1  # image channel: 3
    num_classes = 100  # Cifar-10 total classes (0-9 digits)

    # create Client and model
    return Clients(input_shape=[None, num_input, num_input_channel],
                   num_classes=num_classes,
                   learning_rate=learning_rate,
                   clients_num=num)


  history = {'loss': [], 'acc': []}

  def run_global_test(client, global_vars, test_num):
    acclist = history.get('acc')
    losslist = history.get('loss')
    """ 跑一下测试集，输出ACC和Loss """
    client.set_global_vars(global_vars)
    # acc, loss = client.run_test(test_num)
    scores = client.run_test(test_num)
    acclist.append(scores[1])
    losslist.append(scores[0])
    history['loss'] = losslist
    history['acc'] = acclist
    print("[epoch {}, {} inst] Testing ACC: {:.4f}, Loss: {:.4f}".format(
      ep + 1, test_num, scores[1], scores[0]))


  #### SOME TRAINING PARAMS ####
  CLIENT_NUMBER = 10
  CLIENT_RATIO_PER_ROUND = 0.3  # 每轮挑选clients跑跑看的比例
  epoch = 10  # epoch上限

  totalData = 30426
  #### CREATE CLIENT AND LOAD DATASET ####
  client = buildClients(CLIENT_NUMBER)

  #### BEGIN TRAINING ####
  global_vars = client.get_client_vars()
  clents = {0: 5000, 1: 4000, 2: 3000, 3: 2020, 4: 4020, 5: 3000, 6: 2000, 7: 2500, 8: 2086, 9: 2800}
  for ep in range(epoch):
    print("这是第%d次" % (ep))
    # We are going to sum up active clients' vars at each epoch
    # 用来收集Clients端的参数，全部叠加起来（节约内存）
    client_vars_sum = None

    # Choose some clients that will train on this epoch
    # 随机挑选一些Clients进行训练
    random_clients = client.choose_clients(CLIENT_RATIO_PER_ROUND)
    print(random_clients)
    # Train with these clients
    # 用这些Clients进行训练，收集它们更新后的模型
    for client_id in tqdm(random_clients, ascii=True):
      # Restore global vars to client's model
      # 将Server端的模型加载到Client模型上
      client.set_global_vars(global_vars)

      # train one client
      # 训练这个下标的Client
      chistory = client.train_epoch(cid=client_id)
      # obtain current client's vars
      # 获取当前Client的模型变量值
      current_client_vars = client.get_client_vars()

      # sum it up
      # 把各个层的参数叠加起来
      if client_vars_sum is None:
        client_vars_sum = (pd.Series(current_client_vars) * (clents[client_id] / totalData) * math.exp(
          -chistory.history['val_loss'][0])).tolist()
        # client_vars_sum = (pd.Series(current_client_vars) * (clents[client_id]/totalData)).tolist()
      else:
        current_client_vars = (pd.Series(current_client_vars) * (clents[client_id] / totalData) * math.exp(
          -chistory.history['val_loss'][0])).tolist()
        # current_client_vars = (pd.Series(current_client_vars) * (clents[client_id]/totalData)).tolist()
        for cv, ccv in zip(client_vars_sum, current_client_vars):
          cv += ccv
          # tf.compat.v1.assign_add(cv,ccv)

    # obtain the avg vars as global vars
    # 把叠加后的Client端模型变量 除以 本轮参与训练的Clients数量
    # 得到平均模型、作为新一轮的Server端模型参数
    # global_vars = []
    global_vars = client_vars_sum

    # run test on 1000 instances
    # 跑一下测试集、输出一下
    run_global_test(client, global_vars, test_num=500)

  #### FINAL TEST ####
  run_global_test(client, global_vars, test_num=3381)

  def plot_accuracy(history):
    # plt.plot(history.history['acc'])
    plt.plot(history['acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Test'], loc='upper left')
    img9 = io.BytesIO()
    plt.savefig(img9, format='png')
    img9.seek(0)
    plt.close()
    train_fed_acc_url = base64.b64encode(img9.getvalue()).decode()
    return train_fed_acc_url

  def plot_loss(history):
    # plt.plot(history.history['loss'])
    plt.plot(history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Test'], loc='upper right')
    img10 = io.BytesIO()
    plt.savefig(img10, format='png')
    img10.seek(0)
    plt.close()
    train_fed_loss_url = base64.b64encode(img10.getvalue()).decode()
    return train_fed_loss_url

  train_fed_loss_url = plot_loss(history)
  train_fed_acc_url = plot_accuracy(history)

  color_list = ['red', 'blue', 'gold', 'pink', 'green', 'skyblue', 'teal', 'maroon', 'red', 'coral', 'blue', 'gold']
  layers_name = ['input', "conv_1", "conv_2","dense_1"]
  layers_index = [0, 1, 2, 3]
  fig = plt.figure(figsize=(8, 10))
  fig.tight_layout()
  matplotlib.pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
  plt.title('4WDCNN')
  for i in range(1,4):
    dense1_layer_model = tf.keras.Model(inputs=client.model.get_layer(index=layers_index[0]).output, outputs=client.model.get_layer(index=layers_index[i]).output)
    dense1_output = client.run_predict(3381,dense1_layer_model)
    x = range(20000)
    plt.subplot(4, 1, i)
    plt.plot(x, dense1_output.flatten()[:20000], color=color_list[i])
    plt.ylabel(layers_name[i])

  img2 = io.BytesIO()
  plt.savefig(img2, format='png')
  img2.seek(0)
  plt.close()
  layer_fed_url = base64.b64encode(img2.getvalue()).decode()
  target_path = os.path.join(BASE_DIR, "files/")
  with open(target_path+"layer_fed_url.txt", "w") as f:
    f.write(layer_fed_url)  # 自带文件关闭功能，不需要再写f.close()
  '''混淆矩阵'''
  confusion_matrix_fed_url = client.run_confusion_matrix(3381)
  with open(target_path+"confusion_matrix_fed_url.txt", "w") as f:
    f.write(confusion_matrix_fed_url)  # 自带文件关闭功能，不需要再写f.close()
  with open(target_path+"history_fed.txt", "w") as f:
    f.write(str(history))  # 自带文件关闭功能，不需要再写f.close()
  return render_template('modelTrainfed.html', train_fed_acc_url=train_fed_acc_url,
                         train_fed_loss_url=train_fed_loss_url, testAcc=history['acc'][10])

@app.route("/diagnose/federated")
def diagnosefed():
    return render_template('diagnosefed.html')

@app.route("/diagnose/startfed")
def diagnosestartfed():
    target_path = os.path.join(BASE_DIR, "files/")
    with open(target_path+"confusion_matrix_fed_url.txt", "r") as f:
      plot_diag_fed_url = f.readline()
    with open(target_path+"history_fed.txt", "r") as f:
      history_fed = f.readline()
    history = eval(history_fed)
    return render_template('modeldiagnosefed.html', plot_diag_fed_url=plot_diag_fed_url,testAcc=history['acc'][10])

def unzip_file(zip_src, dst_dir):
    """
    解压zip文件
    :param zip_src: zip文件的全路径
    :param dst_dir: 要解压到的目的文件夹
    :return:
    """
    r = zipfile.is_zipfile(zip_src)
    filelist = []
    if r:
        fz = zipfile.ZipFile(zip_src, "r")
        for file in fz.namelist():
            filelist.append(file)
            fz.extract(file, dst_dir)
    else:
        return "请上传zip类型压缩文件"
    return filelist


@app.route("/upload", methods=["POST"])
def upload():
    if request.method == "GET":
        return render_template("dataprocessfed.html")
    obj = request.files.get("file")

    # 检查上传文件的后缀名是否为zip
    ret_list = obj.filename.rsplit(".", maxsplit=1)
    if len(ret_list) != 2:
        return "请上传zip类型压缩文件"
    if ret_list[1] != "zip":
        return "请上传zip类型压缩文件"

    # 方式三：先保存压缩文件到本地，再对其进行解压，然后删除压缩文件
    file_path = os.path.join(BASE_DIR, "files", obj.filename)  # 上传的文件保存到的路径
    obj.save(file_path)
    target_path = os.path.join(BASE_DIR, "files", "CaseWesternReserveUniversityData")  # 解压后的文件保存到的路径
    filelist = unzip_file(file_path, target_path)
    os.remove(file_path)  # 删除文件
    filename = os.listdir(target_path)
    if isinstance(filelist,list):
        namelist = set(filelist) | set(filename)
    else:
      namelist = filename
    descriptlist = []
    for faultType in namelist:
      faultobj = []
      fault_url = faultType.split('.')
      fault_name = fault_url[0].split('_');
      if len(fault_name) > 2:
        faultDescript = fault_name[0]
        if fault_name[1] == 'Drive':
          faultDescript += '驱动端'
        elif fault_name[1] == 'Fan':
          faultDescript += '风扇端'
        if fault_name[3].startswith('B'):
          faultDescript += '滚动体'
        elif fault_name[3].startswith('I'):
          faultDescript += '内圈'
        elif fault_name[3].startswith('O'):
          faultDescript += '外圈'
        faultDescript += '故障'
      else:
        faultDescript = '正常数据'
      faultobj.append(faultType)
      faultobj.append(faultDescript)
      descriptlist.append(faultobj)
    return render_template('dataprocessfed.html', descriptlist=descriptlist)

@app.route("/dataprocess/faultcurve", methods=["GET"])
def faultcurve():
  faultType = request.args.get("faultType")
  fault_url = faultType.split('.')
  fault_name = fault_url[0].split('_');
  if len(fault_name) >2:
      faultDescript = fault_name[0]
      if fault_name[1] == 'Drive':
        faultDescript += '驱动端'
      elif fault_name[1] == 'Fan':
        faultDescript += '风扇端'
      if fault_name[3].startswith('B'):
        faultDescript += '滚动体'
      elif fault_name[3].startswith('I'):
        faultDescript += '内圈'
      elif fault_name[3].startswith('O'):
        faultDescript += '外圈'
      faultDescript += '故障'
  else:
    faultDescript = '正常数据'

  plt = faultType_data_fed_plot.faultType_plot(fault_url[0])
  img = io.BytesIO()
  plt.savefig(img, format='png')
  img.seek(0)
  faultType_plot_url = base64.b64encode(img.getvalue()).decode()
  plt.close()

  return render_template('faultcurve_plot.html', faultType_plot_url=faultType_plot_url, faultType=fault_url[0],
                         faultDescript=faultDescript)


def unzip_file(zip_src, dst_dir):
    """
    解压zip文件
    :param zip_src: zip文件的全路径
    :param dst_dir: 要解压到的目的文件夹
    :return:
    """
    r = zipfile.is_zipfile(zip_src)
    filelist = []
    if r:
        fz = zipfile.ZipFile(zip_src, "r")
        for file in fz.namelist():
            filelist.append(file)
            fz.extract(file, dst_dir)
    else:
        return "请上传zip类型压缩文件"
    return filelist

# ------------------------------------------边缘计算-起始代码------------------------------------------------
# ------------------------------------------边缘计算-起始代码------------------------------------------------
# ------------------------------------------边缘计算-起始代码------------------------------------------------
# @app.route("/upload", methods=["GET", "POST"])
# def upload():
#     if request.method == "GET":
#         return render_template("dataProcess.html")
#     obj = request.files.get("file")
#     print(obj)  # <FileStorage: "test.zip" ("application/x-zip-compressed")>
#     print(obj.filename)  # test.zip
#     print(obj.stream)  # <tempfile.SpooledTemporaryFile object at 0x0000000004135160>
#
#     # 检查上传文件的后缀名是否为zip
#     ret_list = obj.filename.rsplit(".", maxsplit=1)
#     if len(ret_list) != 2:
#         return "请上传zip类型压缩文件"
#     if ret_list[1] != "zip":
#         return "请上传zip类型压缩文件"
#
#     # 方式一：直接保存文件
#     # obj.save(os.path.join(BASE_DIR, "files", obj.filename))
#
#     # 方式二：保存解压后的文件（原压缩文件不保存）
#     '''target_path = os.path.join(BASE_DIR, "files", str(uuid.uuid4()))
#     shutil._unpack_zipfile(obj.stream, target_path)'''
#
#     # 方式三：先保存压缩文件到本地，再对其进行解压，然后删除压缩文件
#     file_path = os.path.join(BASE_DIR, "files", obj.filename)  # 上传的文件保存到的路径
#     obj.save(file_path)
#     target_path = os.path.join(BASE_DIR, "files", "CaseWesternReserveUniversityData")  # 解压后的文件保存到的路径
#     filelist = unzip_file(file_path, target_path)
#     os.remove(file_path)  # 删除文件
#     if isinstance(filelist,list):
#         return render_template('dataProcess.html', filelist=filelist)
#     else:
#         return filelist
@app.route("/layers_Show")
def layerShow():
    return render_template('layerShowHomePage.html')
@app.route('/layer_4wdcnn_show')
def Branch1_layer():

  TIMESERIES_LENGTH = 100

  f = open("edgedata_0_1_2.txt", "r", encoding='UTF-8')
  lines = f.readlines()
  datas = {}
  fault_type = 'undefined'
  description = ''
  fault_typeList = []
  descriptionList = []
  # fault_types = ['DE_B', 'DE_IR', 'DE_OR', 'FE_B', 'FE_IR', 'FE_B']
  statistics_df = pd.DataFrame(
    columns=['filename', 'fault_type', 'description', 'length', 'DE_min', 'DE_max', 'DE_mean', 'DE_std', 'FE_min',
             'FE_max', 'FE_mean', 'FE_std', 'BA_min', 'BA_max', 'BA_mean', 'BA_std'])
  features = np.empty(shape=(3, 0))


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
    print('Loading data {0} {1} {2}'.format(filename, fault_type, description))
    params = filename.split('_')
    data_no = params[-1]
    # print("///////////////////")
    print(data_no)
    # mat_data = sio.loadmat('./CaseWesternReserveUniversityData/' + filename)

    mat_data = sio.loadmat('./CaseWesternReserveUniversityData/' + filename)
    features_considered = map(lambda key_str: key_str.format(data_no), ["X{0}_DE_time", "X{0}_FE_time", "X{0}_DE_time"])

    current_features = np.array([mat_data[feature].flatten() for feature in features_considered])

    features = np.concatenate((features, current_features), axis=1)  # 列拼接

    data_size = len(mat_data["X{0}_DE_time".format(data_no)])  # current file timeseries length
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
    return (data - mean) / std


  features[0] = normalize(features[0])
  features[1] = normalize(features[1])
  features[2] = normalize(features[2])

  start_index = 0
  for index, row in statistics_df.iterrows():
    fault_type, length = row['fault_type'], row['length']
    current_features = features[:, start_index:start_index + length]
    multidimensional_timeseries = current_features.T
    start_index += length
    data = [multidimensional_timeseries[i:i + TIMESERIES_LENGTH] for i in range(0, length - TIMESERIES_LENGTH, 100)]
    if fault_type not in datas:
      fault_typeList.append(fault_type)
      description = descriptionList[fault_typeList.index(fault_type)]
      # descriptionList.append(description)
      datas[fault_type] = {
        'fault_type': fault_type,
        'description': description,
        'X': np.empty(shape=(0, TIMESERIES_LENGTH, 3))
      }

    datas[fault_type]['X'] = np.concatenate((datas[fault_type]['X'], data))


  # %%
  # random choice
  def choice(dataset, size):
    return dataset[np.random.choice(dataset.shape[0], size, replace=False), :]


  cloud_num = len(fault_typeList)
  label_placeholder = np.zeros(cloud_num, dtype=int)
  x_data = np.empty(shape=(0, TIMESERIES_LENGTH, 3))
  y_data = np.empty(shape=(0, cloud_num), dtype=int)
  DATASET_SIZE = 0
  # BATCH_SIZE = 16
  BATCH_SIZE = 32
  BUFFER_SIZE = 10000
  for index, (key, value) in enumerate(datas.items()):
    sample_size = len(value['X'])
    DATASET_SIZE += sample_size
    print("{0} {1} {2}".format(value['fault_type'], value['description'], sample_size))
    label = np.copy(label_placeholder)
    label[index] = 1  # one-hot encode
    print("========================")
    x_data = np.concatenate((x_data, value['X']))
    labels = np.repeat([label], sample_size, axis=0)
    y_data = np.concatenate((y_data, labels))
  # print(datas.items())
  total_data = [(x_data[i], y_data[i]) for i in range(0, len(x_data))]
  np.random.shuffle(total_data)
  X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
  print(X_train.shape)
  print(y_train.shape)

  # if branchNo=='1':
  inputs = tf.keras.Input(shape=(100, 3))
  y = tf.keras.layers.Conv1D(filters=16, kernel_size=64, strides=16, padding='same', name="conv_1")(inputs)
  y = tf.keras.layers.BatchNormalization(name="batch_1")(y)
  y = tf.keras.layers.Activation('relu', name="relu_1")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_1", data_format="channels_first")(y)
  y = tf.keras.layers.Conv1D(32, kernel_size=3, strides=1, padding='same', name="conv_2")(y)
  y = tf.keras.layers.BatchNormalization(name="batch_2")(y)
  y = tf.keras.layers.Activation('relu', name="relu_2")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_2", padding='valid', data_format="channels_first")(y)
  y = tf.keras.layers.Conv1D(64, kernel_size=3, strides=1, padding='same', name="conv_3")(y)
  y = tf.keras.layers.BatchNormalization(name="batch_3")(y)
  y = tf.keras.layers.Activation('relu', name="relu_3")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_3", padding='valid', data_format="channels_first")(y)
  y = tf.keras.layers.Conv1D(64, kernel_size=3, strides=1, padding='same', name="conv_4")(y)
  y = tf.keras.layers.BatchNormalization(name="batch_4")(y)
  y = tf.keras.layers.Activation('relu', name="relu_4")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_4", padding='valid', data_format="channels_first")(y)

  y = tf.keras.layers.Flatten(name="flat_1")(y)
  y = tf.keras.layers.Dropout(0.2, name="drop_1")(y)
  y = Dense(100, name="dense_1")(y)
  y = Activation("relu", name="relu_6")(y)
  y = Dense(units=10, activation='softmax', name="dense_2")(y)

  model = tf.keras.Model(inputs=inputs, outputs=y, name='model')
  model.compile(optimizer='Adam', loss='categorical_crossentropy',
                metrics=['accuracy'])

  checkpoint = ModelCheckpoint('classification1_10.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
  callbacks_list = [checkpoint]
  model.load_weights('4WDCNN_0_1_2.h5', by_name=True)

  color_list = ['coral', 'blue', 'gold', 'pink', 'green', 'skyblue', 'teal', 'maroon', 'red', 'coral', 'blue', 'gold',
                'pink', 'green', 'skyblue', 'teal', 'maroon', 'red', 'coral', 'blue', 'gold', 'pink', 'green', 'skyblue',
                'teal']
  layers_name = ['input', "conv_1", "conv_2", "conv_3", "conv_4", "dense_1"]
  layers_index = [0, 1, 5, 9, 13, 19]
  # layers_index=[0,4,8,12,18,20]
  iris_im = ''
  x = range(20000)
  plot_num = len(model.layers)
  print(plot_num)
  fig = plt.figure(figsize=(8, 10))
  # fig = plt.figure(1)
  fig.tight_layout()
  matplotlib.pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
  plt.title('4WDCNN')
  for i in range(1,6):
    print(i)
    dense1_layer_model = tf.keras.Model(inputs=inputs, outputs=model.get_layer(index=layers_index[i]).output)

    dense1_output = dense1_layer_model.predict(X_test)
    x = range(20000)
    plt.subplot(6, 1, i + 1)
    plt.plot(x, dense1_output.flatten()[:20000], color=color_list[i])
    plt.ylabel(layers_name[i])

  img = io.BytesIO()
  plt.savefig(img, format='png')
  img.seek(0)
  layer_4WDCNN_url = base64.b64encode(img.getvalue()).decode()
  plt.close()

  # if branchNo == '2':
  inputs = tf.keras.Input(shape=(100, 3))
  y = tf.keras.layers.Conv1D(filters=16, kernel_size=64, strides=16, padding='same', name="conv_1")(inputs)
  y = tf.keras.layers.BatchNormalization(name="batch_1")(y)
  y = tf.keras.layers.Activation('relu', name="relu_1")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_1", data_format="channels_first")(y)
  y = tf.keras.layers.Conv1D(32, kernel_size=3, strides=1, padding='same', name="conv_2")(y)
  y = tf.keras.layers.BatchNormalization(name="batch_2")(y)
  y = tf.keras.layers.Activation('relu', name="relu_2")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_2", padding='valid', data_format="channels_first")(y)
  y = tf.keras.layers.Conv1D(64, kernel_size=3, strides=1, padding='same', name="conv_3")(y)
  y = tf.keras.layers.BatchNormalization(name="batch_3")(y)
  y = tf.keras.layers.Activation('relu', name="relu_3")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_3", padding='valid', data_format="channels_first")(y)
  y = tf.keras.layers.Conv1D(64, kernel_size=3, strides=1, padding='same', name="conv_4")(y)
  y = tf.keras.layers.BatchNormalization(name="batch_4")(y)
  y = tf.keras.layers.Activation('relu', name="relu_4")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_4", padding='valid', data_format="channels_first")(y)
  y = tf.keras.layers.Conv1D(64, kernel_size=3, strides=1, padding='valid', name="conv_5")(y)
  y = tf.keras.layers.BatchNormalization(name="batch_5")(y)
  y = tf.keras.layers.Activation('relu', name="relu_5")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_5", padding='valid', data_format="channels_first")(y)
  y = tf.keras.layers.Flatten(name="flat_1")(y)
  y = tf.keras.layers.Dropout(0.2, name="drop_1")(y)
  y = Dense(100, name="dense_1")(y)
  y = Activation("relu", name="relu_6")(y)
  y = Dense(units=10, activation='softmax', name="dense_2")(y)

  model = tf.keras.Model(inputs=inputs, outputs=y, name='model')
  model.compile(optimizer='Adam', loss='categorical_crossentropy',
                metrics=['accuracy'])

  model.load_weights('5WDCNN_0_1_2.h5', by_name=True)

  layers_name = ['input', "conv_1", "conv_2", "conv_3", "conv_4", "conv_5", "dense_1"]
  color_list = ['coral', 'blue', 'gold', 'pink', 'green', 'skyblue', 'teal', 'maroon', 'red', 'coral', 'blue', 'gold',
                'pink', 'green', 'skyblue', 'teal', 'maroon', 'red', 'coral', 'blue', 'gold', 'pink', 'green',
                'skyblue', 'teal']
  layers_index = [0, 1, 5, 9, 13, 17, 23]
  iris_im = ''
  x = range(20000)
  plot_num = len(model.layers)
  print(plot_num)
  fig = plt.figure(figsize=(8, 10))
  fig.tight_layout()
  matplotlib.pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
  plt.title('5WDCNN')
  for i in range(1,7):
    dense1_layer_model = tf.keras.Model(inputs=inputs, outputs=model.get_layer(index=layers_index[i]).output)
    print(i)
    # print("-----------------------------------------")
    dense1_output = dense1_layer_model.predict(X_test)
    x = range(20000)

    plt.subplot(7, 1, i + 1)
    # plt.subplot(3)
    plt.plot(x, dense1_output.flatten()[:20000], color=color_list[i])
    plt.ylabel(layers_name[i])

  img1 = io.BytesIO()
  plt.savefig(img1, format='png')
  img1.seek(0)

  layer_5WDCNN_url = base64.b64encode(img1.getvalue()).decode()
  plt.close()

  # if branchNo == '3':
  inputs = tf.keras.Input(shape=(100, 3))
  y = tf.keras.layers.Conv1D(filters=16, kernel_size=64, strides=16, padding='same', name="conv_1")(inputs)
  y = tf.keras.layers.BatchNormalization(name="batch_1")(y)
  y = tf.keras.layers.Activation('relu', name="relu_1")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_1", data_format="channels_first")(y)
  y = tf.keras.layers.Conv1D(32, kernel_size=3, strides=1, padding='same', name="conv_2")(y)
  y = tf.keras.layers.BatchNormalization(name="batch_2")(y)
  y = tf.keras.layers.Activation('relu', name="relu_2")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_2", padding='valid', data_format="channels_first")(y)
  y = tf.keras.layers.Conv1D(64, kernel_size=3, strides=1, padding='same', name="conv_3")(y)
  y = tf.keras.layers.BatchNormalization(name="batch_3")(y)
  y = tf.keras.layers.Activation('relu', name="relu_3")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_3", padding='valid', data_format="channels_first")(y)

  y = tf.keras.layers.LSTM(32, activation='tanh', recurrent_activation='hard_sigmoid',
                           kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                           bias_initializer='zeros', return_sequences=True, name="lstm_1")(y)
  y = tf.keras.layers.LSTM(100, activation='tanh', recurrent_activation='hard_sigmoid',
                           kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                           bias_initializer='zeros', return_sequences=True, name="lstm_2")(y)
  y = tf.keras.layers.LSTM(100, activation='tanh', recurrent_activation='hard_sigmoid',
                           kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                           bias_initializer='zeros', return_sequences=True, name="lstm_3")(y)
  y = tf.keras.layers.Flatten(name="flat_1")(y)
  y = tf.keras.layers.Dropout(0.2, name="drop_1")(y)
  y = Dense(100, name="dense_1")(y)
  y = Activation("relu", name="relu_6")(y)
  y = Dense(units=10, activation='softmax', name="dense_2")(y)

  model = tf.keras.Model(inputs=inputs, outputs=y, name='model')
  model.compile(optimizer='Adam', loss='categorical_crossentropy',
                metrics=['accuracy'])

  model.load_weights('3WDCNN_2lstm_0_1_2.h5', by_name=True)
  layers_name = ['input', "conv_1", "conv_2", "conv_3", "lstm_1", "lstm_2", "lstm_3", "dense_1"]
  color_list = ['coral', 'blue', 'gold', 'pink', 'green', 'skyblue', 'teal', 'maroon', 'red', 'coral', 'blue', 'gold',
                'pink', 'green', 'skyblue', 'teal', 'maroon', 'red', 'coral', 'blue']
  layers_index = [0, 1, 5, 9, 13, 14, 15, 18]
  iris_im = ''
  x = range(20000)
  plot_num = len(model.layers)
  print(plot_num)
  fig = plt.figure(figsize=(10, 12))
  fig.tight_layout()
  matplotlib.pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.8)
  for i in range(1,8):
    print(i)
    dense1_layer_model = tf.keras.Model(inputs=inputs, outputs=model.get_layer(index=layers_index[i]).output)
    # print("-----------------------------------------")
    dense1_output = dense1_layer_model.predict(X_test)
    x = range(20000)

    plt.subplot(8, 1, i + 1)
    # plt.subplot(3)
    plt.plot(x, dense1_output.flatten()[:20000], color=color_list[i])
    plt.ylabel(layers_name[i])

  img2 = io.BytesIO()
  plt.savefig(img2, format='png')
  img2.seek(0)

  # layer_CNN_LSTM_url = base64.b64encode(img2.getvalue()).decode()
  layer_CNN_LSTM_url = base64.b64encode(img2.getvalue()).decode()
  plt.close()

  # return render_template('modelLayerShow.html', layer_url=layer_url)
  return render_template('modelLayerShow.html', layer_4WDCNN_url=layer_4WDCNN_url,layer_5WDCNN_url=layer_5WDCNN_url,layer_CNN_LSTM_url=layer_CNN_LSTM_url)

@app.route('/confusion_matrix_HomePage')
def matrix_plot():
  return render_template('matrixShowHomePage.html')

@app.route('/confusion_matrix_4WDCNNshow')
def build_plot():
  img = io.BytesIO()

  TIMESERIES_LENGTH = 100
  f = open("edgedata_0_1_2.txt", "r", encoding='UTF-8')
  lines = f.readlines()
  datas = {}
  fault_type = 'undefined'
  description = ''
  fault_typeList = []
  descriptionList = []
  # fault_types = ['DE_B', 'DE_IR', 'DE_OR', 'FE_B', 'FE_IR', 'FE_B']
  statistics_df = pd.DataFrame(
    columns=['filename', 'fault_type', 'description', 'length', 'DE_min', 'DE_max', 'DE_mean', 'DE_std', 'FE_min',
             'FE_max', 'FE_mean', 'FE_std', 'BA_min', 'BA_max', 'BA_mean', 'BA_std'])
  features = np.empty(shape=(3, 0))


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
    print('Loading data {0} {1} {2}'.format(filename, fault_type, description))
    params = filename.split('_')
    data_no = params[-1]
    # print("///////////////////")
    print(data_no)
    mat_data = sio.loadmat('./CaseWesternReserveUniversityData/' + filename)
    features_considered = map(lambda key_str: key_str.format(data_no), ["X{0}_DE_time", "X{0}_FE_time", "X{0}_DE_time"])
    # print(features_considered.shape)
    # print("------------------------")
    print(features_considered)
    current_features = np.array([mat_data[feature].flatten() for feature in features_considered])
    # print("***************************")
    # print(current_features)
    features = np.concatenate((features, current_features), axis=1)  # 列拼接
    data_size = len(mat_data["X{0}_DE_time".format(data_no)])  # current file timeseries length
    statistics = [filename, fault_type, description, data_size]
    statistics += npstatistics(current_features[0])
    statistics += npstatistics(current_features[1])
    statistics += npstatistics(current_features[2])
    statistics_df.loc[statistics_df.size] = statistics

  f.close()
  print("\nStatistics:")
  print(statistics_df.head())



  features[0] = normalize(features[0])
  features[1] = normalize(features[1])
  features[2] = normalize(features[2])

  start_index = 0
  for index, row in statistics_df.iterrows():
    fault_type, length = row['fault_type'], row['length']
    current_features = features[:, start_index:start_index + length]
    multidimensional_timeseries = current_features.T
    start_index += length
    data = [multidimensional_timeseries[i:i + TIMESERIES_LENGTH] for i in range(0, length - TIMESERIES_LENGTH, 100)]
    if fault_type not in datas:
      fault_typeList.append(fault_type)
      description = descriptionList[fault_typeList.index(fault_type)]
      # descriptionList.append(description)
      datas[fault_type] = {
        'fault_type': fault_type,
        'description': description,
        'X': np.empty(shape=(0, TIMESERIES_LENGTH, 3))
      }

    datas[fault_type]['X'] = np.concatenate((datas[fault_type]['X'], data))

  cloud_num = len(fault_typeList)
  label_placeholder = np.zeros(cloud_num, dtype=int)
  x_data = np.empty(shape=(0, TIMESERIES_LENGTH, 3))
  y_data = np.empty(shape=(0, cloud_num), dtype=int)
  DATASET_SIZE = 0
  for index, (key, value) in enumerate(datas.items()):
    sample_size = len(value['X'])
    DATASET_SIZE += sample_size
    print("{0} {1} {2}".format(value['fault_type'], value['description'], sample_size))
    label = np.copy(label_placeholder)
    label[index] = 1  # one-hot encode
    print("========================")
    # print(value['X'])
    # print(value)
    x_data = np.concatenate((x_data, value['X']))
    labels = np.repeat([label], sample_size, axis=0)
    y_data = np.concatenate((y_data, labels))
  # print(datas.items())
  total_data = [(x_data[i], y_data[i]) for i in range(0, len(x_data))]
  np.random.shuffle(total_data)
  X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
  print(X_train.shape)
  print(y_train.shape)

  inputs = tf.keras.Input(shape=(100, 3))
  y = tf.keras.layers.Conv1D(filters=16, kernel_size=64, strides=16, padding='same', name="conv_1")(inputs)
  y = tf.keras.layers.BatchNormalization(name="batch_1")(y)
  y = tf.keras.layers.Activation('relu', name="relu_1")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_1", data_format="channels_first")(y)
  y = tf.keras.layers.Conv1D(32, kernel_size=3, strides=1, padding='same', name="conv_2")(y)
  y = tf.keras.layers.BatchNormalization(name="batch_2")(y)
  y = tf.keras.layers.Activation('relu', name="relu_2")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_2", padding='valid', data_format="channels_first")(y)
  y = tf.keras.layers.Conv1D(64, kernel_size=3, strides=1, padding='same', name="conv_3")(y)
  y = tf.keras.layers.BatchNormalization(name="batch_3")(y)
  y = tf.keras.layers.Activation('relu', name="relu_3")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_3", padding='valid', data_format="channels_first")(y)
  y = tf.keras.layers.Conv1D(64, kernel_size=3, strides=1, padding='same', name="conv_4")(y)
  y = tf.keras.layers.BatchNormalization(name="batch_4")(y)
  y = tf.keras.layers.Activation('relu', name="relu_4")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_4", padding='valid', data_format="channels_first")(y)
  y = tf.keras.layers.Flatten(name="flat_1")(y)
  y = tf.keras.layers.Dropout(0.2, name="drop_1")(y)
  y = Dense(100, name="dense_1")(y)
  y = Activation("relu", name="relu_6")(y)
  y = Dense(units=10, activation='softmax', name="dense_2")(y)
  # model = tf.keras.Model(inputs=inputs, outputs=y, name='model')
  model = tf.keras.Model(inputs=inputs, outputs=y, name='model')

  model.compile(optimizer='Adam', loss='categorical_crossentropy',
                metrics=['accuracy'])

  start = time.time()
  # model.save_weights('4WDCNN_0_1_2_50.h5')
  model.load_weights('4WDCNN_0_1_2.h5', by_name=True)

  # model = tf.keras.models.load_model('./my_model.h5')
  score = model.evaluate(X_test, y_test, verbose=0)
  print('Test score:', score[0])
  print('Test accuracy:', score[1])
  print(fault_typeList)
  print(descriptionList)
  # np.argmax(model.predict(x_data[104005:104015]))
  indexArray = np.argmax(model.predict(X_test[:10]), axis=1)
  print(indexArray)
  for idx, val in enumerate(indexArray, 1):
    print(idx, val)
    print('故障种类 :', fault_typeList[val])
    print('故障种类名称:', descriptionList[val])
  print('test after load: ', model.predict(X_test[:10]))
  print('test label: ', y_test[:10])
  print(np.argmax(y_test[:10], axis=1))
  print(indexArray)
  end = time.time()
  print('Total time: %f' % (end - start))


  predict_classes = np.argmax(model.predict(X_test), axis=1)
  true_classes = np.argmax(y_test, 1)
  plot_url = confusion_matrix.plot_confusion_matrix(true_classes, predict_classes, save_flg=True)
  plot_branch1_url = plot_url
  testAcc1 = score[1]

  inputs = tf.keras.Input(shape=(100, 3))
  y = tf.keras.layers.Conv1D(filters=16, kernel_size=64, strides=16, padding='same', name="conv_1")(inputs)
  y = tf.keras.layers.BatchNormalization(name="batch_1")(y)
  y = tf.keras.layers.Activation('relu', name="relu_1")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_1", data_format="channels_first")(y)
  y = tf.keras.layers.Conv1D(32, kernel_size=3, strides=1, padding='same', name="conv_2")(y)
  y = tf.keras.layers.BatchNormalization(name="batch_2")(y)
  y = tf.keras.layers.Activation('relu', name="relu_2")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_2", padding='valid', data_format="channels_first")(y)
  y = tf.keras.layers.Conv1D(64, kernel_size=3, strides=1, padding='same', name="conv_3")(y)
  y = tf.keras.layers.BatchNormalization(name="batch_3")(y)
  y = tf.keras.layers.Activation('relu', name="relu_3")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_3", padding='valid', data_format="channels_first")(y)
  y = tf.keras.layers.Conv1D(64, kernel_size=3, strides=1, padding='same', name="conv_4")(y)
  y = tf.keras.layers.BatchNormalization(name="batch_4")(y)
  y = tf.keras.layers.Activation('relu', name="relu_4")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_4", padding='valid', data_format="channels_first")(y)
  y = tf.keras.layers.Conv1D(64, kernel_size=3, strides=1, padding='valid', name="conv_5")(y)
  y = tf.keras.layers.BatchNormalization(name="batch_5")(y)
  y = tf.keras.layers.Activation('relu', name="relu_5")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_5", padding='valid', data_format="channels_first")(y)

  y = tf.keras.layers.Flatten(name="flat_1")(y)
  y = tf.keras.layers.Dropout(0.2, name="drop_1")(y)
  y = Dense(100, name="dense_1")(y)
  y = Activation("relu", name="relu_6")(y)
  y = Dense(units=10, activation='softmax', name="dense_2")(y)
  model = tf.keras.Model(inputs=inputs, outputs=y, name='model')

  model.compile(optimizer='Adam', loss='categorical_crossentropy',
                metrics=['accuracy'])
  start = time.time()
  model.load_weights('5WDCNN_0_1_2.h5', by_name=True)

  score = model.evaluate(X_test, y_test, verbose=0)
  print('Test score:', score[0])
  print('Test accuracy:', score[1])
  print(fault_typeList)
  print(descriptionList)
  # np.argmax(model.predict(x_data[104005:104015]))
  indexArray = np.argmax(model.predict(X_test[:10]), axis=1)
  print(indexArray)
  for idx, val in enumerate(indexArray, 1):
    print(idx, val)
    print('故障种类 :', fault_typeList[val])
    print('故障种类名称:', descriptionList[val])

  print('test after load: ', model.predict(X_test[:10]))
  print('test label: ', y_test[:10])
  print(np.argmax(y_test[:10], axis=1))
  print(indexArray)
  end = time.time()
  print('Total time: %f' % (end - start))

  predict_classes = np.argmax(model.predict(X_test), axis=1)
  true_classes = np.argmax(y_test, 1)
  # confusion_matrix.plot_confusion_matrix(true_classes, predict_classes, save_flg=True)
  plot_branch2_url = confusion_matrix.plot_confusion_matrix(true_classes, predict_classes, save_flg=True)
  plot_branch2_url = plot_branch2_url
  testAcc2 = score[1]

  inputs = tf.keras.Input(shape=(100, 3))
  y = tf.keras.layers.Conv1D(filters=16, kernel_size=64, strides=16, padding='same', name="conv_1")(inputs)
  y = tf.keras.layers.BatchNormalization(name="batch_1")(y)
  y = tf.keras.layers.Activation('relu', name="relu_1")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_1", data_format="channels_first")(y)
  y = tf.keras.layers.Conv1D(32, kernel_size=3, strides=1, padding='same', name="conv_2")(y)
  y = tf.keras.layers.BatchNormalization(name="batch_2")(y)
  y = tf.keras.layers.Activation('relu', name="relu_2")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_2", padding='valid', data_format="channels_first")(y)
  y = tf.keras.layers.Conv1D(64, kernel_size=3, strides=1, padding='same', name="conv_3")(y)
  y = tf.keras.layers.BatchNormalization(name="batch_3")(y)
  y = tf.keras.layers.Activation('relu', name="relu_3")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_3", padding='valid', data_format="channels_first")(y)

  y = tf.keras.layers.LSTM(32, activation='tanh', recurrent_activation='hard_sigmoid',
                           kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                           bias_initializer='zeros', return_sequences=True, name="lstm_1")(y)
  y = tf.keras.layers.LSTM(100, activation='tanh', recurrent_activation='hard_sigmoid',
                           kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                           bias_initializer='zeros', return_sequences=True, name="lstm_2")(y)
  y = tf.keras.layers.LSTM(100, activation='tanh', recurrent_activation='hard_sigmoid',
                           kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                           bias_initializer='zeros', return_sequences=True, name="lstm_3")(y)

  y = tf.keras.layers.Flatten(name="flat_1")(y)
  y = tf.keras.layers.Dropout(0.2, name="drop_1")(y)
  y = Dense(100, name="dense_1")(y)
  y = Activation("relu", name="relu_6")(y)
  y = Dense(units=10, activation='softmax', name="dense_2")(y)
  model = tf.keras.Model(inputs=inputs, outputs=y, name='model')

  model.compile(optimizer='Adam', loss='categorical_crossentropy',
                metrics=['accuracy'])


  start = time.time()

  model.load_weights('3WDCNN_2lstm_0_1_2.h5', by_name=True)

  score = model.evaluate(X_test, y_test, verbose=0)
  print('Test score:', score[0])
  print('Test accuracy:', score[1])
  print(fault_typeList)
  print(descriptionList)
  # np.argmax(model.predict(x_data[104005:104015]))
  indexArray = np.argmax(model.predict(X_test[:10]), axis=1)
  print(indexArray)
  for idx, val in enumerate(indexArray, 1):
    print(idx, val)
    print('故障种类 :', fault_typeList[val])
    print('故障种类名称:', descriptionList[val])
  print('test after load: ', model.predict(X_test[:10]))
  print('test label: ', y_test[:10])
  print(np.argmax(y_test[:10], axis=1))
  print(indexArray)
  end = time.time()
  print('Total time: %f' % (end - start))

  predict_classes = np.argmax(model.predict(X_test), axis=1)
  true_classes = np.argmax(y_test, 1)
  # confusion_matrix.plot_confusion_matrix(true_classes, predict_classes, save_flg=True)
  plot_branch3_url = confusion_matrix.plot_confusion_matrix(true_classes, predict_classes, save_flg=True)

  plot_branch3_url = plot_branch3_url
  testAcc3 = score[1]


  return render_template('plot.html', plot_branch1_url=plot_branch1_url,testAcc1=testAcc1,plot_branch2_url=plot_branch2_url,testAcc2=testAcc2,plot_branch3_url=plot_branch3_url,testAcc3=testAcc3)


@app.route('/dara_Process')
def dara_Process():
  TIMESERIES_LENGTH = 100

  f = open("edgedata_0_1_2.txt", "r", encoding='UTF-8')
  lines = f.readlines()
  datas = {}
  fault_type = 'undefined'
  description = ''
  fault_typeList = []
  descriptionList = []
  # fault_types = ['DE_B', 'DE_IR', 'DE_OR', 'FE_B', 'FE_IR', 'FE_B']
  statistics_df = pd.DataFrame(
    columns=['filename', 'fault_type', 'description', 'length', 'DE_min', 'DE_max', 'DE_mean', 'DE_std', 'FE_min',
             'FE_max', 'FE_mean', 'FE_std', 'BA_min', 'BA_max', 'BA_mean', 'BA_std'])
  features = np.empty(shape=(3, 0))

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
    print('Loading data {0} {1} {2}'.format(filename, fault_type, description))
    params = filename.split('_')
    data_no = params[-1]

    print(data_no)
    mat_data = sio.loadmat('./CaseWesternReserveUniversityData/' + filename)
    features_considered = map(lambda key_str: key_str.format(data_no), ["X{0}_DE_time", "X{0}_FE_time", "X{0}_DE_time"])

    print(features_considered)
    current_features = np.array([mat_data[feature].flatten() for feature in features_considered])

    features = np.concatenate((features, current_features), axis=1)  # 列拼接

    data_size = len(mat_data["X{0}_DE_time".format(data_no)])  # current file timeseries length
    statistics = [filename, fault_type, description, data_size]
    statistics += npstatistics(current_features[0])
    statistics += npstatistics(current_features[1])
    statistics += npstatistics(current_features[2])
    statistics_df.loc[statistics_df.size] = statistics

  f.close()
  print("\nStatistics:")
  print(statistics_df.head())

  features[0] = normalize(features[0])
  features[1] = normalize(features[1])
  features[2] = normalize(features[2])

  start_index = 0
  for index, row in statistics_df.iterrows():
    fault_type, length = row['fault_type'], row['length']
    current_features = features[:, start_index:start_index + length]
    multidimensional_timeseries = current_features.T
    start_index += length
    data = [multidimensional_timeseries[i:i + TIMESERIES_LENGTH] for i in range(0, length - TIMESERIES_LENGTH, 100)]
    if fault_type not in datas:
      fault_typeList.append(fault_type)
      description = descriptionList[fault_typeList.index(fault_type)]
      # descriptionList.append(description)
      datas[fault_type] = {
        'fault_type': fault_type,
        'description': description,
        'X': np.empty(shape=(0, TIMESERIES_LENGTH, 3))
      }

    datas[fault_type]['X'] = np.concatenate((datas[fault_type]['X'], data))

  faultDict = dict(zip(fault_typeList,descriptionList))
# return render_template('dataProcess.html', fault_typeList=fault_typeList , descriptionList=descriptionList)
  print(faultDict)
  # return render_template('dataProcess.html', faultDict=json.loads(faultDict))
  return render_template('dataProcess1.html', faultDict=faultDict)

@app.route('/faultType/<faultType>/', methods=['POST', 'GET'])
# @app.route('/faultType', methods=['POST'])
def faultType(faultType):
# def faultType():
    # print(faultType)
    # faultType = request.form.get('faultType')
    faultDescript=''
    if faultType=='normal':
      fault_url='normal_0_097'
      faultDescript ='正常'

    if faultType == 'DE_B_7':
      fault_url = '12k_Drive_End_B007_0_118'
      faultDescript = '驱动端滚动体故障7'

    if faultType=='DE_B_14':
      fault_url='12k_Drive_End_B014_0_185'
      faultDescript = '驱动端滚动体故障14'

    if faultType=='DE_B_21':
      fault_url='12k_Drive_End_B021_0_222'
      faultDescript = '驱动端滚动体故障21'

    if faultType=='DE_IR_7':
      fault_url='12k_Drive_End_IR007_0_105'
      faultDescript = '驱动端内圈故障7'

    if faultType=='DE_IR_14':
      fault_url='12k_Drive_End_IR014_0_169'
      faultDescript = '驱动端内圈故障14'

    if faultType=='DE_IR_21':
      fault_url='12k_Drive_End_IR021_0_209'
      faultDescript = '驱动端内圈故障21'

    if faultType=='DE_OR_7':
      fault_url='12k_Drive_End_OR007@3_0_144'
      faultDescript = '驱动端外圈故障7'

    if faultType=='DE_OR_14':
      fault_url='12k_Drive_End_OR021@3_0_246'
      faultDescript = '驱动端外圈故障14'

    if faultType=='DE_OR_21':
      fault_url='12k_Drive_End_OR021@6_0_234'
      faultDescript = '驱动端外圈故障21'

    plt=faultType_data_plot.faultType_plot(fault_url)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    faultType_plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return render_template('faultType_plot.html',faultType_plot_url=faultType_plot_url,faultType=faultType,faultDescript=faultDescript)

@app.route('/dataprocessEdge/faultcurve', methods=['GET'])
def faultcurveEdge():
# def faultType():
    # print(faultType)
    faultDescript = request.args.get("faultType")
    # print('------------faultType-------------')
    # print(faultType)
    faultType=''
    fault_url=''
    if faultDescript =='正常':
      fault_url='normal_0_097'
      faultType = 'normal'


    if faultDescript == '驱动端滚动体故障7':
      fault_url = '12k_Drive_End_B007_0_118'
      faultType = 'DE_B_7'

    if faultDescript == '驱动端滚动体故障14' :
      fault_url='12k_Drive_End_B014_0_185'
      faultType = 'DE_B_14'

    if faultDescript == '驱动端滚动体故障21' :
      fault_url='12k_Drive_End_B021_0_222'
      faultType = 'DE_B_21'

    if faultDescript == '驱动端内圈故障7' :
      fault_url='12k_Drive_End_IR007_0_105'
      faultType = 'DE_IR_7'

    if faultDescript == '驱动端内圈故障14':
      fault_url='12k_Drive_End_IR014_0_169'
      faultType = 'DE_IR_14'

    if faultDescript == '驱动端内圈故障21':
      fault_url='12k_Drive_End_IR021_0_209'
      faultType = 'DE_IR_21'

    if faultDescript == '驱动端外圈故障7':
      fault_url='12k_Drive_End_OR007@3_0_144'
      faultType = 'DE_OR_7'

    if faultDescript == '驱动端外圈故障14':
      fault_url='12k_Drive_End_OR021@3_0_246'
      faultType = 'DE_OR_14'

    if faultDescript == '驱动端外圈故障21':
      fault_url='12k_Drive_End_OR021@6_0_234'
      faultType = 'DE_OR_21'

    plt=faultType_data_plot.faultType_plot(fault_url)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    faultType_plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return render_template('faultType_plot.html',faultType_plot_url=faultType_plot_url,faultType=faultType,faultDescript=faultDescript)


@app.route('/cloudDiagnosis', methods=['POST', 'GET'])
def cloudDiagnosis():

  def server_start():
    partition_thrift = thriftpy.load('partition.thrift', module_name='partition_thrift')
    server = make_server(partition_thrift.Partition, Dispacher(), '127.0.0.1', 6000)
    print('Thriftpy server is listening...')

    server.serve()
    # return render_template('Cloud_Edge_diagnosis.html', AccTest=AccTest)


  class Dispacher(object):
    def partition(self, file, ep, pp):
      for filename, content in file.items():
        with open('recv_' + filename, 'wb') as f:
          f.write(content.encode())
      out = infer(SERVER, ep, pp)
      # model_start = time.time()
      # print(model_start)
      if (ep == 1 and pp == 1):
        out.load_weights('./4WDCNN_0_1_2.h5', by_name=True)
      if (ep == 2 and pp == 1):
        out.load_weights('./5WDCNN_0_1_2.h5', by_name=True)
      if (ep == 3 and pp == 1):
        out.load_weights('./3WDCNN_2lstm_0_1_2.h5', by_name=True)
      x_test = testData()
      y_test = labelData()

      out.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

      score = out.evaluate(x_test, y_test, verbose=0)
      print('Test score:', score[0])
      print('Test accuracy:', score[1])
      diagnosisAccTest =score[1]

      indexArray = np.argmax(out.predict(x_test), axis=1)
      # model_end = time.time()
      # print('Total time: %f' %(model_end-model_start))
      pred = str(indexArray[:30])
      print(pred)
      pred=pred+'|'+str(score[1])
      print(pred)
      print('True label is:')
      label = np.argmax(y_test[:30], axis=1)
      print(label)
      return pred
  server_start()

@app.route('/edgeDiagnosis', methods=['POST','GET'])
def edgeDiagnosis():
  FILENAME = 'model.json'

  def client_start():
    partition_thrift = thriftpy.load('partition.thrift', module_name='partition_thrift')
    return make_client(partition_thrift.Partition, '127.0.0.1', 6000,timeout=None)

  def file_info(filename):
    with open(filename, 'rb') as file:
      file_content = file.read()
    return {filename: file_content}


  if request.method == "POST":
      timeToleranceFactor=request.form.get("timeToleranceFactor")
      threshold = timeToleranceFactor

      # get test data
      x_test = testData()
      y_test = labelData()

      start = time.time()

      # get partition point and exit point
      ep, pp = Optimize(threshold)
      print('Branch is %d, and partition point is %d' % (ep, pp))
      out = infer(CLIENT, ep, pp)

      print('Left part of model inference complete.')
      model_json = out.to_json()

      with open("model.json", "w") as json_file:
        json_file.write(model_json)

      client = client_start()
      info = file_info(FILENAME)
      str=client.partition(info, ep, pp)
      pred=str.split('|')[0]
      testAcc=str.split('|')[1]
      print('Predict answer is: ' + pred)
      print(testAcc)
      print(pred[1])
      faultDescript = ''
      if pred[1] == '0':
        faultDescript = '正常'

      if pred[1] == '1':
        faultDescript = '驱动端滚动体故障7'

      if pred[1] == '2':
        faultDescript = '驱动端滚动体故障14'

      if pred[1] == '3':
        faultDescript = '驱动端滚动体故障21'

      if pred[1] == '4':
        faultDescript = '驱动端内圈故障7'

      if pred[1] == '5':
        faultDescript = '驱动端内圈故障14'

      if pred[1] == '6':
        faultDescript = '驱动端内圈故障21'

      if pred[1] == '7':
        faultDescript = '驱动端外圈故障7'

      if pred[1] == '8':
        faultDescript = '驱动端外圈故障14'

      if pred[1] == '9':
        faultDescript = '驱动端外圈故障21'

      end = time.time()
      lastTime=end - start
      print('Total time: %f' % lastTime)

      return render_template('Cloud_Edge_diagnosis1.html',testAcc=testAcc,ep=ep,lastTime=lastTime,faultDescript=faultDescript)
  else:
      return render_template('Cloud_Edge_diagnosis1.html')


@app.route("/trainHomePage")
def trainHomePage():
    return render_template('train.html')

@app.route('/Branch1_train')
def Branch1_Train():
  TIMESERIES_LENGTH = 100

  f = open("edgedata_0_1_2.txt", "r", encoding='UTF-8')
  lines = f.readlines()
  datas = {}
  fault_type = 'undefined'
  description = ''
  fault_typeList = []
  descriptionList = []
  # fault_types = ['DE_B', 'DE_IR', 'DE_OR', 'FE_B', 'FE_IR', 'FE_B']
  statistics_df = pd.DataFrame(
    columns=['filename', 'fault_type', 'description', 'length', 'DE_min', 'DE_max', 'DE_mean', 'DE_std', 'FE_min',
             'FE_max', 'FE_mean', 'FE_std', 'BA_min', 'BA_max', 'BA_mean', 'BA_std'])
  features = np.empty(shape=(3, 0))


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
    print('Loading data {0} {1} {2}'.format(filename, fault_type, description))
    params = filename.split('_')
    data_no = params[-1]
    # print("///////////////////")
    print(data_no)
    mat_data = sio.loadmat('./CaseWesternReserveUniversityData/' + filename)
    features_considered = map(lambda key_str: key_str.format(data_no), ["X{0}_DE_time", "X{0}_FE_time", "X{0}_DE_time"])
    # print(features_considered.shape)
    # print("------------------------")
    print(features_considered)
    current_features = np.array([mat_data[feature].flatten() for feature in features_considered])
    # print("***************************")
    # print(current_features)
    features = np.concatenate((features, current_features), axis=1)  # 列拼接
    # print("+++++++++++++++++++++")
    # print(features)
    # multidimensional_timeseries = np.hstack(current_features)
    data_size = len(mat_data["X{0}_DE_time".format(data_no)])  # current file timeseries length
    statistics = [filename, fault_type, description, data_size]
    statistics += npstatistics(current_features[0])
    statistics += npstatistics(current_features[1])
    statistics += npstatistics(current_features[2])
    statistics_df.loc[statistics_df.size] = statistics

  f.close()
  print("\nStatistics:")
  print(statistics_df.head())


  features[0] = normalize(features[0])
  features[1] = normalize(features[1])
  features[2] = normalize(features[2])

  start_index = 0
  for index, row in statistics_df.iterrows():
    fault_type, length = row['fault_type'], row['length']
    current_features = features[:, start_index:start_index + length]
    multidimensional_timeseries = current_features.T
    start_index += length
    data = [multidimensional_timeseries[i:i + TIMESERIES_LENGTH] for i in range(0, length - TIMESERIES_LENGTH, 100)]
    if fault_type not in datas:
      fault_typeList.append(fault_type)
      description = descriptionList[fault_typeList.index(fault_type)]
      # descriptionList.append(description)
      datas[fault_type] = {
        'fault_type': fault_type,
        'description': description,
        'X': np.empty(shape=(0, TIMESERIES_LENGTH, 3))
      }

    datas[fault_type]['X'] = np.concatenate((datas[fault_type]['X'], data))

  cloud_num = len(fault_typeList)
  label_placeholder = np.zeros(cloud_num, dtype=int)
  x_data = np.empty(shape=(0, TIMESERIES_LENGTH, 3))
  y_data = np.empty(shape=(0, cloud_num), dtype=int)
  DATASET_SIZE = 0
  # BATCH_SIZE = 16
  BATCH_SIZE = 32
  BUFFER_SIZE = 10000
  for index, (key, value) in enumerate(datas.items()):
    sample_size = len(value['X'])
    DATASET_SIZE += sample_size
    print("{0} {1} {2}".format(value['fault_type'], value['description'], sample_size))
    label = np.copy(label_placeholder)
    label[index] = 1  # one-hot encode
    print("========================")
    # print(value['X'])
    # print(value)
    x_data = np.concatenate((x_data, value['X']))
    labels = np.repeat([label], sample_size, axis=0)
    y_data = np.concatenate((y_data, labels))
  # print(datas.items())
  total_data = [(x_data[i], y_data[i]) for i in range(0, len(x_data))]
  np.random.shuffle(total_data)
  X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
  print(X_train.shape)
  print(y_train.shape)
  # training_data=total_data[:len(x_data)*0.8]

  # 将training_data改为total_data,尝试拆分出测试集

  inputs = tf.keras.Input(shape=(100, 3))
  y = tf.keras.layers.Conv1D(filters=16, kernel_size=64, strides=16, padding='same', name="conv_1")(inputs)
  y = tf.keras.layers.BatchNormalization(name="batch_1")(y)
  y = tf.keras.layers.Activation('relu', name="relu_1")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_1", data_format="channels_first")(y)
  y = tf.keras.layers.Conv1D(32, kernel_size=3, strides=1, padding='same', name="conv_2")(y)
  y = tf.keras.layers.BatchNormalization(name="batch_2")(y)
  y = tf.keras.layers.Activation('relu', name="relu_2")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_2", padding='valid', data_format="channels_first")(y)
  y = tf.keras.layers.Conv1D(64, kernel_size=3, strides=1, padding='same', name="conv_3")(y)
  y = tf.keras.layers.BatchNormalization(name="batch_3")(y)
  y = tf.keras.layers.Activation('relu', name="relu_3")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_3", padding='valid', data_format="channels_first")(y)
  y = tf.keras.layers.Conv1D(64, kernel_size=3, strides=1, padding='same', name="conv_4")(y)
  y = tf.keras.layers.BatchNormalization(name="batch_4")(y)
  y = tf.keras.layers.Activation('relu', name="relu_4")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_4", padding='valid', data_format="channels_first")(y)

  y = tf.keras.layers.Flatten(name="flat_1")(y)
  y = tf.keras.layers.Dropout(0.2, name="drop_1")(y)
  y = Dense(100, name="dense_1")(y)
  y = Activation("relu", name="relu_6")(y)
  y = Dense(units=10, activation='softmax', name="dense_2")(y)

  model = tf.keras.Model(inputs=inputs, outputs=y, name='model')

  model.compile(optimizer='Adam', loss='categorical_crossentropy',
                metrics=['accuracy'])

  history = model.fit(X_train, y_train,
      #validation_split=0.25,
  	  validation_split=0.20,
      #epochs=50, batch_size=16,
      epochs=3, batch_size=16,
      verbose=1,callbacks=[])
  start = time.time()
  # model.save_weights('4WDCNN_0_1_2_50.h5')
  # model.load_weights('4WDCNN_0_1_2.h5', by_name=True)

  score = model.evaluate(X_test, y_test, verbose=0)
  print('Test score:', score[0])
  print('Test accuracy:', score[1])
  print(fault_typeList)
  print(descriptionList)
  testAcc1=score[1]
  indexArray = np.argmax(model.predict(X_test[:10]), axis=1)
  print(indexArray)
  for idx, val in enumerate(indexArray, 1):
    print(idx, val)
    print('故障种类 :', fault_typeList[val])
    print('故障种类名称:', descriptionList[val])

  print('test after load: ', model.predict(X_test[:10]))
  print('test label: ', y_test[:10])
  print(np.argmax(y_test[:10], axis=1))
  print(indexArray)
  end = time.time()
  print('Total time: %f' % (end - start))

  def plot_accuracy(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # plt.figure(figsize=(5, 4))
    # plt.show()
    img1_1 = io.BytesIO()
    plt.savefig(img1_1, format='png')
    img1_1.seek(0)
    train_Branch1_acc_url = base64.b64encode(img1_1.getvalue()).decode()
    plt.close()
    return train_Branch1_acc_url


  def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # plt.figure(figsize=(5, 4))
    # plt.show()
    img1_2 = io.BytesIO()
    plt.savefig(img1_2, format='png')
    img1_2.seek(0)
    train_Branch1_loss_url = base64.b64encode(img1_2.getvalue()).decode()
    plt.close()
    return train_Branch1_loss_url

  train_Branch1_loss_url = plot_loss(history)
  train_Branch1_acc_url = plot_accuracy(history)

  inputs = tf.keras.Input(shape=(100, 3))
  y = tf.keras.layers.Conv1D(filters=16, kernel_size=64, strides=16, padding='same', name="conv_1")(inputs)
  # y=tf.keras.layers.Conv1D(32,kernel_size=3, strides=1, padding='same', name="conv_1")(inputs)
  y = tf.keras.layers.BatchNormalization(name="batch_1")(y)
  y = tf.keras.layers.Activation('relu', name="relu_1")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_1", data_format="channels_first")(y)
  y = tf.keras.layers.Conv1D(32, kernel_size=3, strides=1, padding='same', name="conv_2")(y)
  y = tf.keras.layers.BatchNormalization(name="batch_2")(y)
  y = tf.keras.layers.Activation('relu', name="relu_2")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_2", padding='valid', data_format="channels_first")(y)
  y = tf.keras.layers.Conv1D(64, kernel_size=3, strides=1, padding='same', name="conv_3")(y)
  y = tf.keras.layers.BatchNormalization(name="batch_3")(y)
  y = tf.keras.layers.Activation('relu', name="relu_3")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_3", padding='valid', data_format="channels_first")(y)
  y = tf.keras.layers.Conv1D(64, kernel_size=3, strides=1, padding='same', name="conv_4")(y)
  y = tf.keras.layers.BatchNormalization(name="batch_4")(y)
  y = tf.keras.layers.Activation('relu', name="relu_4")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_4", padding='valid', data_format="channels_first")(y)
  y = tf.keras.layers.Conv1D(64, kernel_size=3, strides=1, padding='valid', name="conv_5")(y)
  y = tf.keras.layers.BatchNormalization(name="batch_5")(y)
  y = tf.keras.layers.Activation('relu', name="relu_5")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_5", padding='valid', data_format="channels_first")(y)

  y = tf.keras.layers.Flatten(name="flat_1")(y)
  y = tf.keras.layers.Dropout(0.2, name="drop_1")(y)
  y = Dense(100, name="dense_1")(y)
  y = Activation("relu", name="relu_6")(y)
  y = Dense(units=10, activation='softmax', name="dense_2")(y)
  model = tf.keras.Model(inputs=inputs, outputs=y, name='model')

  model.compile(optimizer='Adam', loss='categorical_crossentropy',
                metrics=['accuracy'])


  history = model.fit(X_train, y_train,
      #validation_split=0.25,
  	validation_split=0.20,
      #epochs=50, batch_size=16,
      epochs=3, batch_size=16,
      verbose=1,callbacks=[])
  start = time.time()

  score = model.evaluate(X_test, y_test, verbose=0)
  print('Test score:', score[0])
  print('Test accuracy:', score[1])
  testAcc2=score[1]
  print(fault_typeList)
  print(descriptionList)
  # np.argmax(model.predict(x_data[104005:104015]))
  indexArray = np.argmax(model.predict(X_test[:10]), axis=1)
  print(indexArray)
  for idx, val in enumerate(indexArray, 1):
    print(idx, val)
    print('故障种类 :', fault_typeList[val])
    print('故障种类名称:', descriptionList[val])

  print('test after load: ', model.predict(X_test[:10]))
  print('test label: ', y_test[:10])
  print(np.argmax(y_test[:10], axis=1))
  print(indexArray)
  end = time.time()
  print('Total time: %f' % (end - start))


  def plot_accuracy(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    img1_1 = io.BytesIO()
    plt.savefig(img1_1, format='png')
    img1_1.seek(0)
    train_Branch2_acc_url = base64.b64encode(img1_1.getvalue()).decode()
    plt.close()
    return train_Branch2_acc_url

  # plot_accuracy(history)

  def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    img1_2 = io.BytesIO()
    plt.savefig(img1_2, format='png')
    img1_2.seek(0)
    train_Branch2_loss_url = base64.b64encode(img1_2.getvalue()).decode()
    plt.close()
    return train_Branch2_loss_url
  # plot_loss(history)
  train_Branch2_loss_url = plot_loss(history)
  train_Branch2_acc_url = plot_accuracy(history)

  inputs = tf.keras.Input(shape=(100, 3))
  y = tf.keras.layers.Conv1D(filters=16, kernel_size=64, strides=16, padding='same', name="conv_1")(inputs)
  y = tf.keras.layers.BatchNormalization(name="batch_1")(y)
  y = tf.keras.layers.Activation('relu', name="relu_1")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_1", data_format="channels_first")(y)
  y = tf.keras.layers.Conv1D(32, kernel_size=3, strides=1, padding='same', name="conv_2")(y)
  y = tf.keras.layers.BatchNormalization(name="batch_2")(y)
  y = tf.keras.layers.Activation('relu', name="relu_2")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_2", padding='valid', data_format="channels_first")(y)
  y = tf.keras.layers.Conv1D(64, kernel_size=3, strides=1, padding='same', name="conv_3")(y)
  y = tf.keras.layers.BatchNormalization(name="batch_3")(y)
  y = tf.keras.layers.Activation('relu', name="relu_3")(y)
  y = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_3", padding='valid', data_format="channels_first")(y)

  y = tf.keras.layers.LSTM(32, activation='tanh', recurrent_activation='hard_sigmoid',
                           kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                           bias_initializer='zeros', return_sequences=True, name="lstm_1")(y)
  y = tf.keras.layers.LSTM(100, activation='tanh', recurrent_activation='hard_sigmoid',
                           kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                           bias_initializer='zeros', return_sequences=True, name="lstm_2")(y)
  y = tf.keras.layers.LSTM(100, activation='tanh', recurrent_activation='hard_sigmoid',
                           kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                           bias_initializer='zeros', return_sequences=True, name="lstm_3")(y)

  y = tf.keras.layers.Flatten(name="flat_1")(y)
  y = tf.keras.layers.Dropout(0.2, name="drop_1")(y)
  y = Dense(100, name="dense_1")(y)
  y = Activation("relu", name="relu_6")(y)
  y = Dense(units=10, activation='softmax', name="dense_2")(y)
  model = tf.keras.Model(inputs=inputs, outputs=y, name='model')

  model.compile(optimizer='Adam', loss='categorical_crossentropy',
                metrics=['accuracy'])

  history = model.fit(X_train, y_train,
      #validation_split=0.25,
  	validation_split=0.20,
      #epochs=50, batch_size=16,
      epochs=2, batch_size=16,
      verbose=1,callbacks=[])
  start = time.time()
  score = model.evaluate(X_test, y_test, verbose=0)
  print('Test score:', score[0])
  print('Test accuracy:', score[1])
  testAcc3=score[1]
  print(fault_typeList)
  print(descriptionList)
  indexArray = np.argmax(model.predict(X_test[:10]), axis=1)
  print(indexArray)
  for idx, val in enumerate(indexArray, 1):
    print(idx, val)
    print('故障种类 :', fault_typeList[val])
    print('故障种类名称:', descriptionList[val])

  print('test after load: ', model.predict(X_test[:10]))
  print('test label: ', y_test[:10])
  print(np.argmax(y_test[:10], axis=1))
  print(indexArray)
  end = time.time()
  print('Total time: %f' % (end - start))


  def plot_accuracy(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    img1_1 = io.BytesIO()
    plt.savefig(img1_1, format='png')
    img1_1.seek(0)
    train_Branch3_acc_url = base64.b64encode(img1_1.getvalue()).decode()
    plt.close()
    return train_Branch3_acc_url
  # plot_accuracy(history)

  def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    img1_2 = io.BytesIO()
    plt.savefig(img1_2, format='png')
    img1_2.seek(0)
    train_Branch3_loss_url = base64.b64encode(img1_2.getvalue()).decode()
    plt.close()
    return train_Branch3_loss_url
  # plot_loss(history)
  train_Branch3_loss_url = plot_loss(history)
  train_Branch3_acc_url = plot_accuracy(history)
  return render_template('modelTrain.html', train_Branch1_acc_url=train_Branch1_acc_url,train_Branch1_loss_url=train_Branch1_loss_url,testAcc1=testAcc1,train_Branch2_loss_url=train_Branch2_loss_url,train_Branch2_acc_url=train_Branch2_acc_url,testAcc2=testAcc2,train_Branch3_loss_url=train_Branch3_loss_url,train_Branch3_acc_url=train_Branch3_acc_url,testAcc3=testAcc3)

# ------------------------------------------边缘计算-结束代码------------------------------------------------
# ------------------------------------------边缘计算-结束代码------------------------------------------------
# ------------------------------------------边缘计算-结束代码------------------------------------------------

if __name__ == '__main__':
    app.run()
