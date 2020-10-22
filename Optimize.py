from config import BRANCH_NUMBER
from Time_Prediction import ServerTime, DeviceTime, partition_point_number, model_size, OutputSizeofPartitionLayer
from functools import reduce

# TODO： B？ 500KB/s for test
B = 4096000

# def str2float(s):
#     def fn(x,y):
#             return x*10+y
#     n=s.find('.')
#     s1=list(map(int,[x for x in s[:n]]))
#     s2=list(map(int,[x for x in s[n+1:]]))
#     return reduce(fn,s1)+reduce(fn,s2)/10**len(s2)

def Optimize(latency_threshold):
    server_time_predictor, device_time_predictor = ServerTime(), DeviceTime()
    for exit_branch in range(BRANCH_NUMBER - 1, -1, -1):
        times = []
        for partition_point in range(partition_point_number[exit_branch]):
            # model load time
            left_model_size = model_size['branch' + str(exit_branch + 1) + '_part' + str(partition_point + 1) + 'L']
            right_model_size = model_size['branch' + str(exit_branch + 1) + '_part' + str(partition_point + 1) + 'R']
            model_load_time = device_time_predictor.device_model_load(left_model_size) + \
                              server_time_predictor.server_model_load(right_model_size)

            # immediate data size(bits)
            outputsize = OutputSizeofPartitionLayer.output_size(exit_branch, partition_point)

            total_time = device_time_predictor.predict_time(exit_branch, partition_point) + \
                         server_time_predictor.predict_time(exit_branch, partition_point) + model_load_time + \
                         outputsize / B
            times.append(total_time)
            print(times)

        # find min latency in this branch
        partition_point = times.index(min(times))

        if total_time < float(latency_threshold):
            return exit_branch + 1, partition_point + 1
    # if no ep and pp can satisfy latency required then return 1, 1
    return 1, 1


if __name__ == '__main__':
    print(Optimize(0.4))
