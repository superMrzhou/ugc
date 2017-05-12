# -*- coding:utf-8 -*-

"""
@version: 1.0
@author: kevin
@license: Apache Licence 
@contact: liujiezhang@bupt.edu.cn
@site: 
@software: PyCharm Community Edition
@file: CNN_helper.py
@time: 17/4/21 17:39
"""
import numpy as np
from itertools import *
def sample(N):
    '''
    伪造数据
    :param N:样本量
    :return:X,Y dtype:np.ndarray
    '''
    x_data,y_data = [],[]
    w = np.random.normal(size=100)
    for i in range(N):
        y = np.zeros(4)
        x = np.random.random(100)
        if np.abs(np.dot(w,x)) <= 3.2:
            y[1:3] = 1
        else:
            y[0],y[-1] = 1,1
        x = ['%.3f'%xx for xx in np.random.random(100)]
        x_data.append(x)
        y_data.append(y)

    return np.array(x_data),np.array(y_data)

def parseLable(file='../docs/CNN/dic_label'):
    '''
    解析层级标签
    :param file:
    :return:
    '''
    id_cate = {}
    G1,G2 = [],[]
    with open(file,'r') as fr:
        for line in fr:
            cont = line.decode('utf-8').split("\t")
            id_cate[int(cont[1])] = cont[0]
            if set(cont[0]) & set(['-','_']):
                G1.append(int(cont[1]))
            else:
                G2.append(int(cont[1]))
    return id_cate,G1,G2

def loadDataSet(file):
    '''
    加载文件数据
    :param file:
    :return:
    '''
    X,Y,raw_data = [],[],[]
    with open(file,'r') as fr:
        for line in fr:
            cont = line.decode('utf-8').split('\t')
            X.append([int(x.split(":")[0]) for x in cont[1].split()])
            Y.append([int(x) for x in cont[0].split()])
            raw_data.append(cont[2])
    return X,Y,raw_data

def transY2Vec(Y,G1,G2):
    '''
    Y转化为向量
    :param Y:
    :param G1:
    :param G2:
    :return:
    '''
    G1.extend(G2)
    Y_vec = np.zeros([len(Y),len(G1)])
    for i, y in enumerate(Y):
        Y_vec[i,y]=1

    return Y_vec


if __name__ == "__main__":
    id_cate,G1,G2 = parseLable()
    print(len(G1),len(G2))
    exit()
    X,Y,raw_data = loadDataSet(file='../docs/CNN/test')
    Y_vec = transY2Vec(Y,G1,G2)