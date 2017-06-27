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

import os
import re
import sys

import yaml
import numpy as np
import json
from math import ceil
from copy import deepcopy
import itertools
from collections import Counter

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adagrad, RMSprop
from keras.metrics import categorical_accuracy

from utils.callbacks import HammingLoss
from utils.metrics import f1_measure, hamming_loss, precision_at_k
from utils.assemble import assemble

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from utils.data_helper import *

reload(sys)
sys.setdefaultencoding('utf8')

#


def get_session(gpu_fraction=0.3):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def train(valid_dataset, test_dataset, params):

    # Assemble and compile the model
    model = assemble('ADIOS', params)
    raw_test_dataset = deepcopy(test_dataset)
    # Prepare embedding layer weights and convert inputs for static model
    model_type = params['iter']['model_type']
    print("Model type is", model_type)
    if model_type == "CNN-non-static" or model_type == "CNN-static":
        embedding_weights = train_word2vec(
            np.vstack((valid_dataset['X'], test_dataset['X'])),
            vocabulary_inv,
            num_features=params['X']['embedding_dim'],
            min_word_count=1,
            context=5)
        if model_type == "CNN-static":
            # train_dataset['X'] = embedding_weights[0][train_dataset['X']]
            test_dataset['X'] = embedding_weights[0][test_dataset['X']]
            valid_dataset['X'] = embedding_weights[0][valid_dataset['X']]

        elif params['iter']['model_type'] == "CNN-non-static":
            embedding_layer = model.get_layer('embedding')
            embedding_layer.set_weights(embedding_weights)
    elif model_type == "CNN-rand":
        embedding_weights = None
    else:
        raise ValueError("Unknown model type")

    # complie model
    model.compile(
        loss={
            'Y0': params['Y0']['loss_func'],
            'Y1': params['Y1']['loss_func']
        },
        loss_weights={
            'Y0': params['Y0']['loss_weight'],
            'Y1': params['Y1']['loss_weight']
        },
        metrics=[categorical_accuracy],
        optimizer=RMSprop(params['iter']['learn_rate']))

    # Make sure checkpoints folder exists
    model_dir = params['iter']['model_dir']
    model_name = params['iter']['model_name']
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # Setup callbacks
    callbacks = [
        HammingLoss({
            'valid': valid_dataset
        }),
        ModelCheckpoint(
            model_dir + model_name,
            monitor='val_hl',
            verbose=0,
            save_best_only=True,
            mode='min'),
        EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min'),
    ]

    # Fit the model to the data
    batch_size = params['iter']['batch_size']
    nb_epoch = params['iter']['epoch']
    train_data_generator = generate_arrays_from_dataset(
        '../docs/CNN/trainString',
        vocabulary,
        Y0Y1,
        params['Y0']['dim'],
        split_tag='@@@',
        lbl_text_index=[0, 1],
        batch_size=batch_size,
        sequence_length=params['X']['sequence_length'])
    # start to train
    model.fit_generator(
        train_data_generator,
        validation_data=({
            'X': valid_dataset['X']
        }, {
            'Y0': valid_dataset['Y0'],
            'Y1': valid_dataset['Y1']
        }),
        steps_per_epoch=1012,
        epochs=nb_epoch,
        callbacks=callbacks,
        verbose=1)

    # Load the best weights
    if os.path.isfile(model_dir + model_name):
        model.load_weights(model_dir + model_name)

    # Fit thresholds
    input_sparse = True if params['iter']['model_type'] != 'CNN-non-static' else None
    # Fit thresholds
    thres_X, thres_Y0, thres_Y1 = [], [], []
    # data_size = 150 * batch_size (2048)
    for _ in range(200):
        x_dict, y_dict = train_data_generator.next()
        thres_X.extend(x_dict['X'].tolist())
        thres_Y0.extend(y_dict['Y0'].tolist())
        thres_Y1.extend(y_dict['Y1'].tolist())
    thres_dataset = {
        'X': np.array(thres_X),
        'Y0': np.array(thres_Y0),
        'Y1': np.array(thres_Y1),
    }
    model.fit_thresholds(
        thres_dataset,
        validation_data=valid_dataset,
        top_k=None,
        alpha=np.logspace(-3, 3, num=10).tolist(),
        verbose=1,
        batch_size=params['iter']['batch_size'],
        input_sparse=input_sparse,
        use_hidden_feature=True,
        vocab_size=params['X']['vocab_size'])

    # Test the model
    probs, preds = model.predict_combine(
        test_dataset,
        verbose=1,
        batch_size=params['iter']['batch_size'],
        use_hidden_feature=True, )

    targets_all = np.hstack([test_dataset[k] for k in ['Y0', 'Y1']])
    preds_all = np.hstack([preds[k] for k in ['Y0', 'Y1']])
    # save predict sampless
    save_predict_samples(
        raw_test_dataset, test_dataset, preds_all, save_num=2000)
    for i in range(100):
        print('\n')
        print(' '.join([vocabulary_inv[ii]
                        for ii in raw_test_dataset['X'][i]]))
        print(np.where(targets_all[i] == 1))
        print(' '.join([Y0Y1[ii] for ii in np.where(targets_all[i] == 1)[0]]))
        print(np.where(preds_all[i] == 1))
        print(' '.join([Y0Y1[ii] for ii in np.where(preds_all[i] == 1)[0]]))

    # print('start calculate confuse matix....')
    # get_confuse(test_dataset, preds, 'Y0')
    # get_confuse(test_dataset, preds, 'Y1')

    hl = hamming_loss(test_dataset, preds)
    f1_macro = f1_measure(test_dataset, preds, average='macro')
    f1_micro = f1_measure(test_dataset, preds, average='micro')
    f1_samples = f1_measure(test_dataset, preds, average='samples')
    p_at_1 = precision_at_k(test_dataset, probs, K=1)
    p_at_3 = precision_at_k(test_dataset, probs, K=3)
    p_at_5 = precision_at_k(test_dataset, probs, K=5)

    for k in ['Y0', 'Y1', 'all']:
        print
        print("Hamming loss (%s): %.2f" % (k, hl[k]))
        print("F1 macro (%s): %.4f" % (k, f1_macro[k]))
        print("F1 micro (%s): %.4f" % (k, f1_micro[k]))
        print("F1 sample (%s): %.4f" % (k, f1_samples[k]))
        print("P@1 (%s): %.4f" % (k, p_at_1[k]))
        print("P@3 (%s): %.4f" % (k, p_at_3[k]))
        print("P@5 (%s): %.4f" % (k, p_at_5[k]))

    t_recall, t_precision, t_f1 = recall_precision_f1(targets_all, preds_all)
    # t_recall, t_precision = all_recall_precision(test_dataset['Y1'], preds['Y1'])
    print('total recall : %.4f' % t_recall)
    print('total precision : %.4f' % t_precision)
    print('total f1 : %.4f\n' % t_f1)

    g1_recall, g1_precision, g1_f1 = recall_precision_f1(
        test_dataset['Y0'], preds['Y0'])
    print('G1 recall : %.4f' % g1_recall)
    print('G1 precision : %.4f' % g1_precision)
    print('G1 f1 : %.4f\n' % g1_f1)

    g2_recall, g2_precision, g2_f1 = recall_precision_f1(
        test_dataset['Y1'], preds['Y1'])
    print('G2 recall : %.4f' % g2_recall)
    print('G2 precision : %.4f' % g2_precision)
    print('G2 f1 : %.4f\n' % g2_f1)


def save_predict_samples(raw_test_dataset,
                         test_dataset,
                         preds_all,
                         save_num=None):
    with open('../docs/CNN/test_pre_result.txt', 'w') as f:
        save_num = len(
            test_dataset['X']) if save_num is None else int(save_num)
        gt_matrix = np.concatenate(
            [test_dataset['Y0'], test_dataset['Y1']], axis=-1)
        for i in range(save_num):
            text = ' '.join(
                [vocabulary_inv[ii] for ii in raw_test_dataset['X'][i]])
            gt = ' '.join([Y0Y1[ii] for ii in np.where(gt_matrix[i] == 1)[0]])
            pre = ' '.join([Y0Y1[ii] for ii in np.where(preds_all[i] == 1)[0]])
            f.write('%s@@@%s@@@%s\n' % (gt, pre, text))


def all_recall_precision(Y1_true, preds):
    '''
    以二级标签构建一级标签，计算整体准确率
    '''
    gt_lbls_n, tp_lbls_n, pr_lbls_n = 0., 0., 0.
    for i in range(len(Y1_true)):
        gt_ind = np.where(Y1_true[i] == 1)[0]
        gt_lbl = [re.split('_|-', Y1[ii])[0]
                  for ii in gt_ind] + [Y1[ii] for ii in gt_ind]

        pred_ind = np.where(preds[i] == 1)[0]
        pred_lbl = [re.split('_|-', Y1[ii])[0]
                    for ii in pred_ind] + [Y1[ii] for ii in pred_ind]

        gt_lbls_n += len(set(gt_lbl))
        pr_lbls_n += len(set(pred_lbl))
        tp_lbls_n += len(set(gt_lbl) & set(pred_lbl))
    return tp_lbls_n / gt_lbls_n, tp_lbls_n / pr_lbls_n


def get_confuse(data, pred, kw):
    y_true, y_pre = [], []
    _cate = Y0 if kw == 'Y0' else Y1
    for i in range(len(data[kw])):
        y_true.append([_cate[ii] for ii in np.where(data[kw][i] == 1)[0]])
        y_pre.append([_cate[ii] for ii in np.where(pred[kw][i] == 1)[0]])
    confuse_dict = ml_confuse(y_true, y_pre)
    with open('../docs/CNN/%s_confuse' % kw, 'w') as f:
        for lbl, c_dict in confuse_dict.items():
            c_sort = sorted(c_dict.items(), key=lambda d: d[1], reverse=True)
            cont = ' '.join(map(lambda x: '%s:%s' % x, c_sort)) + '\n'
            f.write('%s\t%s' % (lbl, cont))


def y2vec(y, cate_id, cateIds_list):

    res = np.zeros((len(y), len(cateIds_list)))
    for i, yy in enumerate(y):
        res[i][[cateIds_list.index(lbl) for lbl in yy]] = 1
    return res


def recall_precision_f1(y_true, y_pre):

    gt_lbls_n, tp_lbls_n, pr_lbls_n = 0., 0., 0.
    for i in range(len(y_true)):
        gt_ind = np.where(y_true[i] == 1)[0]
        pred_ind = np.where(y_pre[i] == 1)[0]
        gt_lbls_n += len(gt_ind)
        pr_lbls_n += len(pred_ind)
        tp_lbls_n += len(set(gt_ind) & set(pred_ind))
    print('tp:%s\nprecision_dem:%s\nrecall_dem:%s' % (tp_lbls_n, pr_lbls_n,
                                                      gt_lbls_n))
    recall = tp_lbls_n / gt_lbls_n
    acc = tp_lbls_n / pr_lbls_n
    f1 = 2 * recall * acc / (recall + acc)
    return recall, acc, f1


def y2list(y):

    # y = [yy[0].strip('\n').split(' ') for yy in y]
    return [
        list(set([re.split('-|_', lbl)[0] for lbl in yy] + yy)) for yy in y
    ]


def y2list_and_g1(y):

    # y = [yy[0].strip('\n').split('&') for yy in y]
    return [
        list(set(['%s_G1' % re.split('-|_', lbl)[0] for lbl in yy] + yy))
        for yy in y
    ]


def get_Y0_and_Y1(file_path):
    with open(file_path, 'r') as f_cate:
        Y0, Y1 = [], []
        for line in f_cate:
            if re.search('-|_', line.split('\t')[0]):
                Y0.append(line.strip('\n').split('\t')[1])
            else:
                Y1.append(line.strip('\n').split('\t')[1])
    return Y0, Y1


def filter_data(x, y):
    res_x, res_y = [], []
    for i, yy in enumerate(y):
        temp_y = filter(lambda lbl: '其他' not in lbl and '新闻' not in lbl, yy)
        if len(temp_y) > 1:  # and not (len(temp_y) == 1 and 'G1' in temp_y[0]):
            res_x.append(x[i])
            res_y.append(temp_y)
        if i % 2000 == 0:
            print i
    return res_x, res_y


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
    print(len(vocabulary_inv), len(vocabulary))

    # load tst_file
    tst_texts, tst_labels = load_data_and_labels(
        '../docs/CNN/testString', lbl_text_index=[0, 1], split_tag='@@@')

    # category
    # 一级原始类目
    G0 = [
        yy[0]
        for yy in load_data_and_labels(
            '../docs/CNN/cate_Y0', split_tag='\t', lbl_text_index=[1, 0])[0]
    ]
    # 二级原始类目
    G1 = [
        yy[0]
        for yy in load_data_and_labels(
            '../docs/CNN/cate_Y1', split_tag='\t', lbl_text_index=[1, 0])[0]
    ] + G0
    # 是否需要反转标签
    Y1 = ['%s_G1' % yy for yy in G0]
    Y0 = G1
    Y0Y1 = Y0 + Y1
    print(len(Y0Y1))

    # params
    nb_features = len(vocabulary_inv)
    nb_labels = len(Y0 + Y1)
    nb_labels_Y0 = len(Y0)
    nb_labels_Y1 = len(Y1)

    # load params config
    params = yaml.load(open('../docs/configs/adios.yaml'))
    params['X']['vocab_size'] = len(vocabulary)
    params['Y0']['dim'] = nb_labels_Y0
    params['Y1']['dim'] = nb_labels_Y1
    print json.dumps(params, indent=4)

    # print('train data size : %d , test data size : %d' % (len(trn_labels),
    #                                                       len(tst_labels)))

    # set gpu_option
    KTF.set_session(get_session(gpu_fraction=params['iter']['gpu_fraction']))

    ratio = 0.02
    valid_N = int(ratio * len(tst_texts))
    # train_dataset just input raw data
    # train_dataset = {'X': trn_texts, 'Y': trn_labels}
    # valid_dataset should transform
    test_X, test_Y0, test_Y1 = [], [], []
    for i in range(len(tst_texts)):
        x, y0, y1 = process_line(
            tst_texts[i],
            tst_labels[i],
            vocabulary,
            Y0Y1,
            nb_labels_Y0,
            sequence_length=256)
        test_X.append(x)
        test_Y0.append(y0)
        test_Y1.append(y1)
    valid_dataset = {
        'X': np.array(test_X)[:valid_N, :],
        'Y0': np.array(test_Y0)[:valid_N, :],
        'Y1': np.array(test_Y1)[:valid_N, :],
    }
    test_dataset = {
        'X': np.array(test_X)[valid_N:, :],
        'Y0': np.array(test_Y0)[valid_N:, :],
        'Y1': np.array(test_Y1)[valid_N:, :],
    }
    # start train
    train(valid_dataset, test_dataset, params)
