import csv
import numpy as np
import matplotlib.pyplot as plt

'''
现在已知一定的和时间有关的序列
(如金融序列等)
我们可以通过rnn网络,通过以往的数据,来对未来走势进行预测
虽然在真正金融中进行预测可能会有问题,但是我们可以先拿一些简单的数据来看看rnn如何做时间序列的预测
'''

#将模型读取进来,返回标准化后的数据
def load_series(filename, series_idx=1):
    try:
        with open(filename) as csvfile: #打开文件
            csvreader = csv.reader(csvfile) #用csv类reader()方法读取文件
            data = [float(row[series_idx]) for row in csvreader if len(row) > 0] #第二列数据
            normalized_data = (data - np.mean(data)) / np.std(data) #标准化
        return normalized_data
    except IOError: #IO错误,返回空
        return None

#将数据切分成训练集和测试集
def split_data(data, percent_train=0.80):
    num_rows = len(data)
    train_data, test_data = [], []
    for idx, row in enumerate(data):
        if idx < num_rows * percent_train:
            train_data.append(row) #训练集
        else:
            test_data.append(row) #测试集
    return train_data, test_data


if __name__=='__main__':
    timeseries = load_series('international-airline-passengers.csv')
    print('数据形状 : ', np.shape(timeseries))

    plt.figure()
    plt.plot(timeseries)
    plt.show()

