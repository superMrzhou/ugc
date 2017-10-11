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
import tensorflow as tf
from keras import backend as K
from utils.assemble import HISO
from utils.data_helper import build_data_cv


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
    number_example = len(eval_data)
    eval_loss, eval_acc, eval_counter = 0., 0., 0
    for start, end in zip(
            range(0, number_example, batch_size),
            range(batch_size, number_example, batch_size)):
        curr_eval_loss = sess.run(
            model.loss,
            feed_dict={
                model.inputs: [hml.vec for hml in eval_data],
                model.Y0: [hml.top_label for hml in eval_data],
                model.Y1: [hml.bottom_label for hml in eval_data],
            })

        eval_loss += curr_eval_loss
        eval_counter += 1

        return eval_loss / eval_counter


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
        for epoch in range(3):
            for start, end in zip(
                    range(0, number_of_training_data, batch_size),
                    range(batch_size, number_of_training_data, batch_size)):

                inputs = [hml.vec for hml in train_datas[start:end]]
                Y0 = [hml.top_label for hml in train_datas[start:end]]
                Y1 = [hml.bottom_label for hml in train_datas[start:end]]

                trn_loss, _ = sess.run(
                    [hiso.loss, hiso.train_op],
                    feed_dict={hiso.inputs: inputs,
                               hiso.Y0: Y0,
                               hiso.Y1: Y1})
                if (end / batch_size) % 4 == 0:
                    tst_loss = do_eval(sess, hiso, test_datas, batch_size)
                    print('epoch: {}, train loss: {}, test loss: {}'.format(
                        epoch, trn_loss, tst_loss))
