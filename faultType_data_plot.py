import scipy.io as sio
import matplotlib.pyplot as plt


def get_data(filename):
    filename = '.\CaseWesternReserveUniversityData\{}.mat'.format(filename)
    data = sio.loadmat(filename)
    return data

def get_driver_end_data(filename,point_number=10000):
    data = get_data(filename)
    file_number = filename.split('_')[-1]
    key = 'X{}_DE_time'.format(file_number.zfill(3))
    return data[key].flatten()[:point_number]





def faultType_plot(fault_url):
    point_number = 2000
    filenames_12k = [fault_url]
    driver_end_data = list(map(lambda filename:get_driver_end_data(filename,point_number),filenames_12k))

    x=range(point_number)

    fig=plt.figure(figsize=(7,2))
    # plt.rcParams['fig.figsize'] = (18.0, 4.0)
    fig.tight_layout()
    plt.subplot(1,1,1)
    plt.plot(x,driver_end_data[0])

    #plt.title('Normal')
    plt.ylabel('Acceleration')
    # plt.show()
    return plt