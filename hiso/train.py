# -*- coding:utf-8 -*-
"""
@version: 1.0
@author: kevin
@license: Apache Licence
@contact: liujiezhang@bupt.edu.cn
@site:
@software: PyCharm Community Edition
@file: adios_train.py
@time: 17/05/03 17:39
"""
import json
import os
import time
from math import ceil

import numpy as np
from sklearn import linear_model as lm
from utils.metrics import (Average_precision, Coverage, Hamming_loss,
                           One_error, Ranking_loss, Construct_thresholds)


import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from config import params
from utils import hiso
from utils.data_helper import *


def train(dataloader):
    '''
    训练模型入口
    '''
    # build model
    timestamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    loss_key = [
        'Hamming_loss', 'One_error', 'Ranking_loss', 'Coverage',
        'Average_precision'
    ]

    # 保存最优模型
    # model_dir = params.model_dir + time.strftime("%Y-%m-%d-%H:%M:%S",
    #                                                time.localtime())
    # s.mkdir(model_dir)
    # model_name = model_dir + '/' + params.model_name
    # build model
    model = hiso.HISO(params)
    # learning rate
    lr = params.lr
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()
    model.train()

    for epoch in range(params.epochs):
        for batch_idx, samples in enumerate(dataloader, 0):
            v_word = Variable(samples['word_vec'])
            v_pos = Variable(samples['pos_vec'])

            v_auix_label = Variable(samples['bottom_label'])
            v_final_label = Variable(samples['top_label'])

            final_probs, auxi_probs = model(v_word, v_pos)
            # autograd optim
            optimizer.zero_grad()
            loss = criterion(final_probs, v_final_label) + criterion(auxi_probs, v_auix_label)
            loss.backward()
            optimizer.step()

            # evaluate train model


            if batch_idx % params.log_interval == 0:
                print(epoch, batch_idx, loss.data[0])
            
            

if __name__ == '__main__':
    # load params
    trainset = UGCDataset(file_path='../docs/data/HML_JD_ALL.new.dat', voc_path='../docs/data/voc.json', pos_path='../docs/data/pos.json')
    
    train_loader = DataLoader(trainset,
            batch_size=48,
            shuffle=True,
            num_workers=1)
    
    train(train_loader)
