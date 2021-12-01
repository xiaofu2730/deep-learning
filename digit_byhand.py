'''
Description:numpy实现全连接神经网络,并训练,加载手写数字识别模型
Author: Chen Xiaofu
Date: 2021-11-28 12:50:45
LastEditTime: 2021-12-02 00:17:44
LastEditors: Chen XiaoFu
Contact me: 2020302191130@whu.edu.cn
'''
#官方库
import numpy as np
import random
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
#个人数据加载包
import minist_loader.mnist_loader


def sigmoid(z):
    """ 激活函数为sigmoid """
    return 1.0/(1.0+np.exp(-z))

def sigmoid_diff(z):
    """ 激活函数的微分 """
    return sigmoid(z)*(1-sigmoid(z))

def get_random_bias(sizes):
    """
    随机生成每一层神经元的阈值
    注意第一层没有bias
    每一层随机生成的bias 是一个正态分布
    sizes是一个存储每层神经元个数的列表
    """
    bias = []
    for k in sizes[1:]:
        bias.append(np.random.randn(k, 1))
    return bias

def get_random_weights(sizes):
    """
    随机生成每一层神经元与前一层神经元连接权重
    """
    weights = []
    for pre_size, post_size in zip(sizes[:-1], sizes[1:]):
        weights.append(np.random.randn(post_size, pre_size))
    return weights

def get_split_mini_batch(train_data, mini_batch_size, total_len):
    """
    将整个训练集分成多个mini_batch
    """
    batches = []
    for i in range(0, total_len, mini_batch_size):
        batches.append(train_data[i:i+mini_batch_size])
    return batches

def load_pretrained_fnc_model(hprams_json):
    net = FullConnectedNet(begin_train=False)
    with open(hprams_json+'.json', "r", encoding='utf-8') as f:
        hparams_dict = json.load(f)
    bias = []
    weights = []
    temp_bias = hparams_dict['best_bias']
    temp_weights = hparams_dict['best_weights']
    for i in temp_bias:
        bias.append(np.array(i))
    for j in temp_weights:
        weights.append(np.array(j))
    net._bias = bias
    net._weights = weights
    net._num_layers=len(bias)+1
    return net
    # TODO transfer type from list to ndarry

class FullConnectedNet(object):
    """
    自定义全连接神经网络
    """
    def __init__(self, sizes=[],begin_train=True):
        """
        ——输入:
        sizes:装有每一层神经元个数的列表
        begin_train:是否随机初始化神经网络参数
        ——初始化属性:
        _num_layers:整数,神经网络层数
        _bias:列表，装有阈值
        _weights:列表,装有权重
        """
        if begin_train:
            self._num_layers = len(sizes)
            self._bias = get_random_bias(sizes=sizes)
            self._weights = get_random_weights(sizes=sizes)
        else:
            self._num_layers = 0
            self._bias = []
            self._weights = []
        self.best_model = {}

    def feedforward(self, a):
        """
        前向传播:通过迭代实现
        """
        for b, w in zip(self._bias,self._weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def SGD(self, train_data,epoch_num,mini_batch_size,learning_rate,test_data=None,save_haprams=False,json_name='hparams'):
        """
        ——输入:
        train_data:预处理过的数据
        epoch_num:周期数
        mini_batch_size:小的数据集大小
        learning_rate:神经网络学习率
        save_haprams:布尔型,保存参数
        json_name:字符串,保存的json文件名
        ——主要训练流程:
        1) 每一个大周期,打乱train_data,并重新分成多个batch
        2) 调用update_mini_batch方法,更新神经网络参数
        3) 输出每个周期成功率
        4) 记录当前最高成功率
        5) 如果要求保存参数,则保存到相应json文件中
        """
        if test_data:
            n_test=len(test_data)
        n=len(train_data)
        highest_rate = 0
        for i in range(epoch_num):
            random.shuffle(train_data)
            mini_batches = get_split_mini_batch(train_data=train_data,mini_batch_size=mini_batch_size,total_len=n)

            for mini_batch in tqdm(mini_batches):
                self.update_mini_batch(mini_batch, learning_rate-0.005*i)

            if test_data:
                print("Epoch %d: Accuracy rate: %.2f%% learning rate: %.3f"%(i, self.evaluate(test_data)/n_test*100,learning_rate-0.005*i))
                if self.evaluate(test_data)/n_test*100 >highest_rate:
                    self.best_model = self.save_best_model()
            else:
                print("Epoch {} finished!".format(i))
        if save_haprams:
            self.save_haprams_in_json(json_name)
            print("The hparams has been dumped in to {} successfully".format(json_name))


    def update_mini_batch(self, mini_batch,learning_rate):
        """
        每一次小的训练,更新参数

        """
        nabla_b = [np.zeros(b.shape) for b in self._bias]
        nabla_w = [np.zeros(w.shape) for w in self._weights]

        for x, y in mini_batch:
            delta_b,delta_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b,delta_b)]
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w,delta_w)] 


        self._weights = [w-(learning_rate/len(mini_batch))*nw for w,nw in zip(self._weights,nabla_w)]
        self._bias = [b-(learning_rate/len(mini_batch))*nb for b,nb in zip(self._bias,nabla_b)]

    def backprop(self, x,y):
        """
        反向传播算法
        ——输入:
        x:输入向量 长度为28*28的列向量
        y:正确的概率分布 长度为10的列向量
        """
        nabla_b = [np.zeros(b.shape) for b in self._bias]
        nabla_w = [np.zeros(w.shape) for w in self._weights]

        # feedforward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self._bias,self._weights):
            z = np.dot(w,activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward
        delta = self.cost_derivative(activations[-1],y)*sigmoid_diff(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta,activations[-2].transpose())
        for l in range(2, self._num_layers):
            z = zs[-l]
            sp = sigmoid_diff(z)
            delta = np.dot(self._weights[-l+1].transpose(),delta)*sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta,activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        返回测试集中正确分类的数目
        """
        test_results = [(np.argmax(self.feedforward(x)),y) for (x,y) in test_data]

        return sum(int(x ==y) for (x,y) in test_results)

    def save_best_model(self):
        """
        将当前的权值和阈值整理为json可以接受的词典并返回
        """
        best_model = {}
        bias = []
        weights = []
        for i in self._bias:
            lis = []
            for j in i:
                lis.append(list(j))
            bias.append(lis)
        for j in self._weights:
            temp = []
            for i in j:
                temp.append(list(i))
            weights.append(temp)
        best_model['best_bias'] = bias
        best_model['best_weights'] = weights
        return best_model

    def cost_derivative(self, output_activations,y):
        """
        损失函数的偏微分
        """
        return (output_activations-y)

    def save_haprams_in_json(self, name):
        """
        ——输入:
        name:字符串,要保存的json文件名
        清空json文件后,以字典的形式写入参数到json文件中
        """
        with open(name+'.json', "w",encoding='utf-8') as f:
            f.seek(0)
            f.truncate()
        with open(name+'.json', "w",encoding='utf-8') as f:
            json.dump(self.best_model, f,indent=4)

if __name__ =="__main__":
    train_data,validation_data,test_data = minist_loader.mnist_loader.load_data_wrapper()
    fc=FullConnectedNet([784,30,30,10])
    fc.SGD(train_data,20,10,1.0,test_data=test_data,save_haprams=True)
    net = load_pretrained_fnc_model(hprams_json='hparams')
    cnt=0
    wrong_cnt=0
    for (x, y) in validation_data:
        cnt+=1
        if np.argmax(net.feedforward(x))!=y:
            wrong_cnt+=1
        # print(np.argmax(net.feedforward(x)), y)
        # x = x.reshape(28,28)
        # plt.imshow(x)
        # plt.colorbar()
        # plt.xticks(())
        # plt.yticks(())
        # plt.show()
        time.sleep(1)
        #break
    print("{}: wrong_cnt:{} accuracy:{:.2f}%".format(cnt,wrong_cnt,1-wrong_cnt/cnt))
