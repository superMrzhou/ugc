# -*- coding:utf-8 -*-
"""
@version: 1.0
@author: kevin
@license: Apache Licence
@contact: liujiezhang@bupt.edu.cn
@site:
@software: Atom
@file: gru_cnn_train.py
@time: 17/06/30 15:31
"""
import json
import os
import sys
import re

import keras.backend.tensorflow_backend as KTF
import numpy as np
import tensorflow as tf
import yaml
from utils.data_helper_attention_cnn_gru import (load_data_and_labels,
                                                 load_labels_title_content,
                                                 process_line)
from utils.gru_cnn_model import GRU_CNN_Attention

reload(sys)
sys.setdefaultencoding('utf8')


def get_session(gpu_fraction=0.3):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


if __name__ == '__main__':

    # load vocabulary
    vocabulary_inv = list(
        set([
            wd[0]
            for wd in load_data_and_labels(
                '../docs/CNN/featureMap', lbl_text_index=[1, 0])[0]
        ]))
    # add <PAD/>
    vocabulary_inv.insert(0, '<PAD/>')
    vocabulary = {v: i for i, v in enumerate(vocabulary_inv)}
    print('vocabulary length : %s' % len(vocabulary))

    # load tst_file
    val_labels, val_titles, val_contents = load_labels_title_content(
        '../docs/CNN/valString_title', lbl_text_index=[0, 1, 2], split_tag='@@@')
    # category
    # 所有类目
    cate = [
        yy[0]
        for yy in load_data_and_labels(
            '../docs/CNN/categoryMap', split_tag='\t', lbl_text_index=[1, 0])[0]
    ]
    # 一级原始类目
    G0 = filter(lambda yy: not re.search('-|_', yy), cate)
    # 二级原始类目
    G1 = filter(lambda yy: re.search('-|_', yy), cate)
    # 是否需要反转标签
    Y1 = ['%s_G1' % yy for yy in G0]
    Y0 = G0 + G1
    Y0Y1 = Y0 + Y1
    print(len(Y0Y1))

    # params
    nb_features = len(vocabulary_inv)
    nb_labels = len(Y0 + Y1)
    nb_labels_Y0 = len(Y0)
    nb_labels_Y1 = len(Y1)

    # load params config
    params = yaml.load(open('../docs/configs/title_content.yaml'))
    params['Y0']['dim'] = nb_labels_Y0
    params['Y1']['dim'] = nb_labels_Y1
    params['title_layer']['embedding_dic_dim'] = len(vocabulary)
    params['content_layer']['embedding_dic_dim'] = len(vocabulary)
    print json.dumps(params, indent=4)

    # print('train data size : %d , test data size : %d' % (len(trn_labels),
    #                                                       len(tst_labels)))

    # set gpu_option
    KTF.set_session(get_session(gpu_fraction=params['iter']['gpu_fraction']))

    test_title, test_content, test_Y0, test_Y1 = [], [], [], []
    for i in range(len(val_titles)):
        x_title, x_content, y0, y1 = process_line(
            val_titles[i],
            val_contents[i],
            val_labels[i],
            vocabulary,
            Y0Y1,
            nb_labels_Y0,
            title_sequence_length=params['title_layer']['sequence_length'],
            content_sequence_length=params['content_layer']['sequence_length'])
        test_title.append(x_title)
        test_content.append(x_content)
        test_Y0.append(y0)
        test_Y1.append(y1)
    valid_dataset = {
        'title': np.array(test_title),
        'content': np.array(test_content),
        'Y0': np.array(test_Y0),
        'Y1': np.array(test_Y1),
    }

    # complie model
    gru_cnn_model = GRU_CNN_Attention(params)
    gru_cnn_model.train(
        '../docs/CNN/trainString_title',
        '../docs/CNN/testString_title',
        vocabulary_inv,
        Y0Y1,
        test_data_file='../docs/CNN/testString_title',
        valid_data=valid_dataset)
