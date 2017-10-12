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
import time

import numpy as np
import tensorflow as tf
from keras import backend as K
from utils.assemble import HISO
from utils.data_helper import build_data_cv
from utils.metrics import (Average_precision, Coverage, F1_measure,
                           Hamming_loss, One_error, Ranking_loss)


def padding_sentence(hmlObjectList, sequence_len):
    '''
    padding_sentence
    '''
    print('start padding sentence...')
    pad_data = [('<s>', '<s>')]

    def padding(hml):
        if hml.sentence_len <= sequence_len:
            hml.content += pad_data * (sequence_len - hml.sentence_len)
        else:
            hml.content = hml.content[:sequence_len]
        return hml

    return list(map(padding, hmlObjectList))


def word2index(hmlObjectList, vocab_word2index):
    '''
    transform word to index
    '''
    print('start transform word to index...')

    def transform(hml):
        wds = [wd for wd, _ in hml.content]
        hml.vec = [vocab_word2index[wd] for wd in wds]
        return hml

    return list(map(transform, hmlObjectList))


def do_eval(sess, model, eval_data, batch_size):
    '''
    eval test data for moedel.
    '''
    number_of_data = len(eval_data)
    Y0_labels, Y1_labels, Y0_probs, Y1_probs = [], [], [], []
    eval_loss, eval_cnt = 0., 0.
    for start, end in zip(
            range(0, number_of_data, batch_size),
            range(batch_size, number_of_data, batch_size)):
        eval_Y0_labels = [hml.top_label for hml in eval_data[start:end]]
        eval_Y1_labels = [hml.bottom_label for hml in eval_data[start:end]]

        curr_loss, eval_Y0_probs, eval_Y1_probs = sess.run(
            [model.loss, model.Y0_preds, model.Y1_preds],
            feed_dict={
                model.inputs: [hml.vec for hml in eval_data[start:end]],
                model.Y0: eval_Y0_labels,
                model.Y1: eval_Y1_labels,
            })
        eval_loss += curr_loss
        eval_cnt += 1

        Y0_labels.extend(eval_Y0_labels)
        Y1_labels.extend(eval_Y1_labels)
        Y0_probs.extend(eval_Y0_probs)
        Y1_probs.extend(eval_Y1_probs)

    # evaluation metrics
    Y0_labels = np.array(Y0_labels)
    Y1_labels = np.array(Y1_labels)
    Y0_probs = np.array(Y0_probs)
    Y1_probs = np.array(Y1_probs)
    # probs to predict label over thresholds
    # TODO: fit_threshold automatally
    Y0_preds = Y0_probs >= 0.3
    Y1_preds = Y1_probs >= 0.3

    loss_dict = {'eval_loss': eval_loss / eval_cnt, 'Y0': {}, 'Y1': {}}
    # use eval
    func_eval = [
        'Hamming_loss', 'One_error', 'Ranking_loss', 'Coverage',
        'Average_precision'
    ]
    for func in func_eval:
        if func == 'Hamming_loss':
            loss_dict['Y0'][func] = eval(func)(Y0_labels, Y0_preds)
            loss_dict['Y1'][func] = eval(func)(Y1_labels, Y1_preds)
        else:
            loss_dict['Y0'][func] = eval(func)(Y0_labels, Y0_probs)
            loss_dict['Y1'][func] = eval(func)(Y1_labels, Y1_probs)

    return loss_dict


if __name__ == '__main__':
    datas, vocab = build_data_cv('../docs/HML_JD_ALL.new.dat', cv=5)
    # bulid vocab
    vocab_wds = ['<s>'] + list(vocab.keys())
    vocab_word2index = {wd: i for i, wd in enumerate(vocab_wds)}
    vocab_index2word = {i: wd for wd, i in vocab_word2index.items()}
    # split test and train
    test_datas = filter(lambda data: data.cv_n == 1, datas)
    train_datas = filter(lambda data: data.cv_n != 1, datas)
    # process data
    # 1. padding sentence
    # 2. tranform to word index
    test_datas = padding_sentence(test_datas, sequence_len=100)
    train_datas = padding_sentence(train_datas, sequence_len=100)

    test_datas = word2index(test_datas, vocab_word2index)
    train_datas = word2index(train_datas, vocab_word2index)
    print(train_datas[0])

    print('train dataset: {}'.format(len(train_datas)))
    print('test dataset: {}'.format(len(test_datas)))

    number_of_training_data = len(train_datas)
    batch_size = 32
    # build model
    timestamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    loss_key = [
        'Hamming_loss', 'One_error', 'Ranking_loss', 'Coverage',
        'Average_precision'
    ]

    with tf.Session() as sess:
        K.set_session(sess)
        hiso = HISO(
            sentence_len=100,
            Y0_dim=3,
            Y1_dim=6,
            vocab_size=len(vocab_wds),
            embed_size=100)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        with open('../docs/%s.log' % timestamp, 'w') as f:
            for epoch in range(3):
                for start, end in zip(
                        range(0, number_of_training_data, batch_size),
                        range(batch_size, number_of_training_data,
                              batch_size)):

                    inputs = [hml.vec for hml in train_datas[start:end]]
                    Y0 = [hml.top_label for hml in train_datas[start:end]]
                    Y1 = [hml.bottom_label for hml in train_datas[start:end]]

                    trn_loss, _ = sess.run(
                        [hiso.loss, hiso.train_op],
                        feed_dict={
                            hiso.inputs: inputs,
                            hiso.Y0: Y0,
                            hiso.Y1: Y1
                        })
                    if (end / batch_size) % 5 == 0:
                        loss_dict = do_eval(sess, hiso, test_datas, batch_size)

                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S",
                                                  time.localtime())
                        str_loss = '{}:  epoch: {} eval_loss: {}\n'.format(
                            timestamp, epoch, loss_dict['eval_loss'])
                        print(str_loss)
                        f.writelines(str_loss)
                        for key in loss_key:
                            f.writelines('Y0_{}:\t{}\tY1_{}:\t{}\n'.format(
                                key, loss_dict['Y0'][key], key, loss_dict['Y1']
                                [key]))
                        f.writelines('\n')