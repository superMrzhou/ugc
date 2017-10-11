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
    for tst in test_datas[:20]:
        print(tst)
