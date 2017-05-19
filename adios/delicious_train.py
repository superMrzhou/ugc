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
reload(sys)
sys.setdefaultencoding('utf8')

import yaml
import numpy as np
from copy import deepcopy

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adagrad

from utils.callbacks import HammingLoss
from utils.metrics import f1_measure, hamming_loss, precision_at_k
from utils.assemble import assemble
from utils.data_helper import *
from utils.delicious import Delicious as Dataset


def train(train_dataset, valid_dataset, test_dataset, params):

    # Assemble and compile the model
    model = assemble('ADIOS', params)
    raw_test_dataset = deepcopy(test_dataset)
    # Prepare embedding layer weights and convert inputs for static model
    model_type = params['iter']['model_type']
    print("Model type is", model_type)
    if model_type == "CNN-non-static" or model_type == "CNN-static":
        embedding_weights = train_word2vec(np.vstack((train_dataset['X'], valid_dataset['X'], test_dataset['X'])), vocabulary_inv,
                                           num_features=params['X']['embedding_dim'],
                                           min_word_count=1,
                                           context=5)
        if model_type == "CNN-static":
            train_dataset['X'] = embedding_weights[0][train_dataset['X']]
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
    model.compile(loss={'Y0': 'binary_crossentropy',
                        'Y1': 'categorical_crossentropy'},
                  loss_weights={'Y0': 0.4,
                                'Y1': 0.6},
                  metrics=['accuracy'],
                  optimizer=Adagrad(1e-1))

    # Make sure checkpoints folder exists
    model_dir = params['iter']['model_dir']
    model_name = params['iter']['model_name']
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # Setup callbacks
    callbacks = [
        HammingLoss({'valid': valid_dataset}),
        ModelCheckpoint(model_dir + model_name,
                        monitor='val_hl',
                        verbose=0,
                        save_best_only=True,
                        mode='min')
        #EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min'),
    ]  # TODO 早停止参数需要进一步确定 (zhangliujie)

    # Fit the model to the data
    batch_size = params['iter']['batch_size']
    nb_epoch = params['iter']['epoch']

    # start to train
    model.fit(x=train_dataset["X"],
              y=[train_dataset['Y0'], train_dataset['Y1']],
              validation_data=(valid_dataset["X"], [
                               valid_dataset["Y0"], valid_dataset["Y1"]]),
              batch_size=batch_size,
              epochs=nb_epoch,
              callbacks=callbacks,
              verbose=1)

    # Load the best weights
    if os.path.isfile(model_dir + model_name):
        model.load_weights(model_dir + model_name)

    # Fit thresholds
    model.fit_thresholds(train_dataset, validation_data=valid_dataset,
                         alpha=np.logspace(-3, 3, num=20).tolist(), verbose=1)

    # Test the model
    probs, preds = model.predict_threshold(test_dataset, verbose=1)

    targets_all = np.hstack([test_dataset[k] for k in ['Y0', 'Y1']])
    preds_all = np.hstack([preds[k] for k in ['Y0', 'Y1']])
    for i in range(30):
        print('\n')
        print(' '.join([vocabulary_inv[ii]
                        for ii in raw_test_dataset['X'][i]]))
        print(' '.join([id_cate[Y0Y1[ii]]
                        for ii in np.where(np.concatenate([test_dataset['Y0'], test_dataset['Y1']], axis=-1)[i] == 1)[0]]))
        print(np.where(targets_all[i] == True))
        print(' '.join([id_cate[Y0Y1[ii]]
                        for ii in np.where(targets_all[i] == True)[0]]))
        print(np.where(preds_all[i] == True))
        print(' '.join([id_cate[Y0Y1[ii]]
                        for ii in np.where(preds_all[i] == True)[0]]))
    # exit()
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


def y2vec(y, cate_id, cateIds_list):

    res = np.zeros((len(y), len(cateIds_list)))
    for i, yy in enumerate(y):
        res[i][[cateIds_list.index(cate_id[lbl]) for lbl in yy]] = 1
    return res


def y2list(y):

    y = [yy[0].strip('\n').split('&') for yy in y]
    return [list(set([re.split('-|_', lbl)[0] for lbl in yy])) + yy for yy in y]


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
        if temp_y:
            res_x.append(x[i])
            res_y.append(temp_y)
    return res_x, res_y


if __name__ == '__main__':

    # Load the datasets
    labels_order = 'random'
    train = Dataset(which_set='train', stop=80.0,
                    labels_order=labels_order)
    valid = Dataset(which_set='train', start=80.0,
                    labels_order=labels_order)
    test  = Dataset(which_set='test',
                    labels_order=labels_order)

    print(test[:10])
    exit()
    nb_label_splits = 100
    nb_features = train.X.shape[1]
    nb_labels = train.y.shape[1]
    nb_labels_Y0 = 65 #(nb_labels // nb_label_splits) * 18
    nb_labels_Y1 = nb_labels - nb_labels_Y0

    # Specify datasets in the format of dictionaries
    train_dataset = {'X': train.X,
                     'Y0': train.y[:,:nb_labels_Y0],
                     'Y1': train.y[:,nb_labels_Y0:]}
    valid_dataset = {'X': valid.X,
                     'Y0': valid.y[:,:nb_labels_Y0],
                     'Y1': valid.y[:,nb_labels_Y0:]}
    test_dataset = {'X': test.X,
                    'Y0': test.y[:,:nb_labels_Y0],
                    'Y1': test.y[:,nb_labels_Y0:]}

    # params
    nb_features = len(vocabulary_inv)
    nb_labels = len(Y0 + Y1)
    nb_labels_Y0 = len(Y0)
    nb_labels_Y1 = len(Y1)

    trn_text = np.array(trn_text)
    tst_text = np.array(tst_text)
    print('train data size : %d , test data size : %d' %
          (len(trn_labels), len(tst_labels)))
    print('X sequence_length is : %d , Y dim : %d' %
          (trn_text.shape[1], trn_labels.shape[1]))
    # load params config
    params = yaml.load(open('../docs/configs/adios.yaml'))
    params['X']['sequence_length'] = trn_text.shape[1]
    params['X']['vocab_size'] = len(vocabulary)
    params['Y0']['dim'] = nb_labels_Y0
    params['Y1']['dim'] = nb_labels_Y1
    print(params)
    # Specify datasets in the format of dictionaries
    # trn_labels = trn_labels[:50000]
    # trn_text = trn_text[:50000]
    # tst_labels = tst_labels[:5000]
    # tst_text = tst_text[:5000]
    ratio = 0.2
    valid_N = int(ratio * tst_text.shape[0])
    train_dataset = {'X': trn_text,
                     'Y0': trn_labels[:, :nb_labels_Y0],
                     'Y1': trn_labels[:, nb_labels_Y0:]}
    valid_dataset = {'X': tst_text[:valid_N],
                     'Y0': tst_labels[:valid_N, :nb_labels_Y0],
                     'Y1': tst_labels[:valid_N, nb_labels_Y0:]}
    test_dataset = {'X': tst_text[valid_N:],
                    'Y0': tst_labels[valid_N:, :nb_labels_Y0],
                    'Y1': tst_labels[valid_N:, nb_labels_Y0:]}

    # start train
    train(train_dataset, valid_dataset, test_dataset, params)
