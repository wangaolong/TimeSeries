import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import data_loader
import matplotlib.pyplot as plt

#这个类就是从rnn_enxample中原封不动搬来的
class SeriesPredictor:

    def __init__(self, input_dim, seq_size, hidden_dim):
        # Hyperparameters
        self.input_dim = input_dim
        self.seq_size = seq_size
        self.hidden_dim = hidden_dim #隐层神经元100

        # Weight variables and input placeholders
        #预测出来的结果是1个数字,所以这里shape是[hidden_dim, 1]
        self.W_out = tf.Variable(tf.random_normal([hidden_dim, 1]), name='W_out')
        self.b_out = tf.Variable(tf.random_normal([1]), name='b_out')
        self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim])
        self.y = tf.placeholder(tf.float32, [None, seq_size])

        # Cost optimizer
        self.cost = tf.reduce_mean(tf.square(self.model() - self.y))
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)

        # Auxiliary ops
        self.saver = tf.train.Saver() #模型保存器saver

    def model(self):
        """
        :param x: inputs of size [T, batch_size, input_size]
        :param W: matrix of fully-connected output layer weights
        :param b: vector of fully-connected output layer biases
        """
        cell = rnn.BasicLSTMCell(self.hidden_dim)
        #outputs的shape是(?, 5, 100),也就是?个样本,5个时序,100个隐层神经元
        outputs, states = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)
        num_examples = tf.shape(self.x)[0] #样本数
        #W shape是(100, 1),经过扩充之后是(1, 100, 1),经过tile之后是(?, 100, 1)
        #其中?是样本数量
        W_repeated = tf.tile(tf.expand_dims(self.W_out, 0), [num_examples, 1, 1])
        #W_repeated的shape是(?, 100, 1)
        #则out是(?, 5, 1)
        out = tf.matmul(outputs, W_repeated) + self.b_out
        out = tf.squeeze(out) #去掉多余维度
        #out代表所有样本,所有5个时序的输出值
        return out #tf的op,shape是(?, 5)

    def train(self, train_x, train_y, test_x, test_y):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            '''
            我们这里不指定迭代次数
            而是指定一个max_patience为3
            
            在while循环中进行一次又一次迭代(训练)
            每迭代100次,就用测试集跑一次cost,得到一个error
            用min_test_err保存最小error
            
            if test_err < min_test_err:
            这行代码是检测每次跑test的error是否小于min_test_error
            只要条件成立,就把这次实验的error赋值给最小error
            同时将patience置为max_patience,也就是3
            
            如果条件不成立,则patience -= 1
            说明这次的模型还没上次好呢,也即是说,模型开始震荡
            如果某一次检验之后patience等于0,则跳出while循环
            '''
            max_patience = 3
            patience = max_patience
            min_test_err = float('inf') #指定最小的error是无限
            step = 0
            while patience > 0:
                #进行一次迭代
                _, train_err = sess.run([self.train_op, self.cost], feed_dict={self.x: train_x, self.y: train_y})
                if step % 100 == 0:
                    test_err = sess.run(self.cost, feed_dict={self.x: test_x, self.y: test_y})
                    print('step: {}\t\ttrain err: {}\t\ttest err: {}'.format(step, train_err, test_err))
                    if test_err < min_test_err:
                        min_test_err = test_err
                        patience = max_patience
                    else:
                        patience -= 1
                step += 1
            save_path = self.saver.save(sess, './model/') #保存的路径
            print('Model saved to {}'.format(save_path))

    def test(self, sess, test_x):
        tf.get_variable_scope().reuse_variables()
        self.saver.restore(sess, './model/') #读取模型
        #这里run的是所有样本,5个时序分别得出的预测值
        output = sess.run(self.model(), feed_dict={self.x: test_x})
        return output #返回预测结果

def plot_results(train_x, predictions, actual, filename):
    plt.figure()
    #训练集点数
    num_train = len(train_x)
    #前若干个点是训练数据
    plt.plot(list(range(num_train)), train_x, 'b', label='training data')
    #训练数据后若干个点是预测数据
    plt.plot(list(range(num_train, num_train + len(predictions))), predictions, 'r', label='predicted')
    #训练数据后若干个点是实际数据
    plt.plot(list(range(num_train, num_train + len(actual))), actual, 'g', label='test data')
    plt.legend() #添加图例
    if filename is not None:
        plt.savefig(filename) #存储图像
    else:
        plt.show() #展示图像
if __name__ == '__main__':
    '''
    一次扔进入一个数字(input_dim=1),一共5个时序,一共扔进去五个数字
    '''
    seq_size = 5
    predictor = SeriesPredictor(input_dim=1, seq_size=seq_size, hidden_dim=100)
    #加载数据
    data = data_loader.load_series('international-airline-passengers.csv')
    #将加载进来的数据集切分成训练集和测试集,0.8 : 0.2
    #type是list
    train_data, test_data = data_loader.split_data(data)
    #使用断言来验证数据集是否被很好的切割
    assert(len(train_data) + len(test_data) == len(data))

    train_x, train_y = [], [] #训练数据和训练标签
    for i in range(len(train_data) - seq_size): #这里-1浪费掉了最后1个数字
        '''
        将数据5个5个切片,每个切片都是一个样本
        每个样本都是5个数,一次扔进去一个数,一共5个时序扔进去
        
        train_data[i : i + seq_size]是把一维的数据train_data按照5个数5个数切分
        切分出来的也是一维的,shape是(5, )
        使用expand_dims来添加一个维度变为(5, 1),转换成list类型
        再添加到train_x中
        
        train_x的shape是(?, 5, 1),其中?是样本数,5是时序数,1是每个时序的输入
        '''
        train_x.append(np.expand_dims(train_data[i:i+seq_size], axis=1).tolist())
        '''
        train_y是数据是i+1到i+seq_size+1,也就是和train_x的每个数据刚好错开一个数据
        用这样的标签进行训练,最终我们就能训练出一个网络
        它能根据以往数据的规律预测下一个数据
        
        train_y的shape是(?, 5)
        '''
        train_y.append(train_data[i+1:i+seq_size+1])

    test_x, test_y = [], [] #测试数据和测试标签
    for i in range(len(test_data) - seq_size):
        test_x.append(np.expand_dims(test_data[i:i + seq_size], axis=1).tolist())
        test_y.append(test_data[i + 1:i + seq_size + 1])

    #将训练和测试相关的数据放进去进行训练,每训练一定次数就测试一下
    predictor.train(train_x, train_y, test_x, test_y)

    #开始测试
    with tf.Session() as sess:
        #test()方法的返回值,是所有样本,5个时序分别得出的预测值
        '''
        这里说一下为什么最后[:, 0]
        predicted_vals是所有测试集样本的5个时序分别的预测结果
        而这5个时序的预测结果其实就是一个样本的5个数字分别的预测结果
        由于数据切片时步长为1
        (比如1,2,3,4,5,6,7)就会被切成(1,2,3,4,5),(2,3,4,5,6),(3,4,5,6,7)
        所以第一个测试样本的5个时序其实就是(1,2,3,4,5)分别对下一个数字的预测结果
        因此只需要[:, 0]即可取出1这个位置对下一个数字的预测结果
        而(2,3,4,5,6)取出的就是2这个位置对下一个数字的预测结果
        这样把所有行都取出来之后,拼凑在一起的就是所有样本中相邻位置的预测及结果
        
        但是这里得到的并不是所有的预测结果,假设测试样本7个数字
        但是最后得到的predicted_vals的长度肯定不到7
        因为我们5个数字切分为一个样本,每个样本预测时只[:,0]找到第一个数字
        所以最后的预测结果肯定不是7个,而是要少几个
        '''
        predicted_vals = predictor.test(sess, test_x)[:,0]
        print('predicted_vals', np.shape(predicted_vals))
        #训练数据,预测数据,测试数据
        #这个预测结果predicted_vals是按照测试集来一个一个预测的
        plot_results(train_data, predicted_vals, test_data, 'predictions.png')

        prev_seq = train_x[-1]
        predicted_vals = []
        for i in range(20): #循环连续预测20次
            '''
            由于切割的时候,训练集后面紧贴着测试集
            因此这里prev_seq是训练集的最后一个数据
            使用训练集最后一个数据作为接下来连续预测的初值
            
            train_x的shape是(?, 5, 1),那么train_x[-1]就是最后一个样本的(5, 1)
            但是test()函数需要传进去三维的数据,所以对于一个list用中括号括起来,
            相当于是升一个维度,shape变为[1, 5, 1]
            传进test()中进行测试,由于传进去的只有一个样本(1, 5, 1)
            因此返回的next_seq也是这一个样本,5个时序的结果(5, ) (这里去除掉了多余的维度)
            '''
            next_seq = predictor.test(sess, [prev_seq])
            predicted_vals.append(next_seq[-1]) #将预测结果添加到列表中
            '''
            这里解释一下为什么是把next_seq的最后一个值合并进去
            因为一次扔进去的样本是(1,2,3,4,5),而预测后的结果是(2,3,4,5,6)
            所以最后一个数字就是6,而prev_seq[1:]就是(2,3,4,5)
            将6合并到(2,3,4,5)后面就得到了下一次的数据(2,3,4,5,6)
            
            prev_seq[1:]的shape : (4, 1)
            next_seq[-1]的shape : (1, )
            所以使用vstack进行纵向合并
            '''
            prev_seq = np.vstack((prev_seq[1:], next_seq[-1])) #从第二个数开始和next_seq合并
        #这个预测结果是基于训练集进行连续预测
        plot_results(train_data, predicted_vals, test_data, 'hallucinations.png')