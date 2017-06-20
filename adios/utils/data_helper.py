# -*- coding:utf-8 -*-
"""
@version: 1.0
@author: kevin
@license: Apache Licence
@contact: liujiezhang@bupt.edu.cn
@site:
@software: PyCharm Community Edition
@file: data_helper.py
@time: 17/05/03 12:01
"""

import re
import os
import itertools
import xlrd
import time
import numpy as np
from collections import defaultdict
from collections import Counter
from gensim.models import word2vec
from os.path import join, exists, split


def train_word2vec(sentence_matrix,
                   vocabulary_inv,
                   num_features=100,
                   min_word_count=1,
                   context=10):
    """
    Trains, saves, loads Word2Vec model
    Returns initial weights for embedding layer.

    inputs:
    sentence_matrix # int matrix: num_sentences x max_sentence_len
    vocabulary_inv  # list
    num_features    # Word vector dimensionality
    min_word_count  # Minimum word count
    context         # Context window size
    """
    model_dir = os.path.abspath('../docs') + '/model/w2v_matrix'
    model_name = "{:d}features_{:d}minwords_{:d}context".format(
        num_features, min_word_count, context)
    model_name = join(model_dir, model_name)
    if exists(model_name):
        embedding_model = word2vec.Word2Vec.load(model_name)
        print('Load existing Word2Vec model \'%s\'' % split(model_name)[-1])
    else:
        # Set values for various parameters
        num_workers = 2  # Number of threads to run in parallel
        downsampling = 1e-3  # Downsample setting for frequent words

        # Initialize and train the model
        print('Training Word2Vec model...')
        sentences = [[vocabulary_inv[w] for w in s] for s in sentence_matrix]
        embedding_model = word2vec.Word2Vec(
            sentences,
            workers=num_workers,
            size=num_features,
            min_count=min_word_count,
            window=context,
            sample=downsampling)

        # If we don't plan to train the model any further, calling
        # init_sims will make the model much more memory-efficient.
        embedding_model.init_sims(replace=True)

        # Saving the model for later use. You can load it later using
        # Word2Vec.load()
        if not exists(model_dir):
            os.mkdir(model_dir)
        print('Saving Word2Vec model \'%s\'' % split(model_name)[-1])
        embedding_model.save(model_name)

    # add unknown words
    embedding_weights = [
        np.array([
            embedding_model[w] if w in embedding_model else np.random.uniform(
                -0.25, 0.25, embedding_model.vector_size)
            for w in vocabulary_inv
        ])
    ]
    return embedding_weights


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?！？，。；’‘“”’\'\`]", " ", string)

    return string.strip().lower()


def load_data_and_labels(file_path,
                         split_tag='\t',
                         lbl_text_index=[0, 1],
                         is_shuffle=False):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """

    # Load data from files
    with open(file_path, 'r') as f:
        # parse label
        labels = [
            data.strip('\n').split(split_tag)[lbl_text_index[0]]
            for data in f
        ]
    with open(file_path, 'r') as f:
        # parse text
        texts = [
            data.strip('\n').split(split_tag)[lbl_text_index[1]]
            for data in f
        ]

    # Split by words
    # texts = [clean_str(sent) for sent in texts]
    texts = [filter(lambda a: a != '', s.split(" ")) for s in texts]
    # support multi-label
    labels = [filter(lambda a: a != '', s.split(" ")) for s in labels]
    if is_shuffle:
        ind = np.arange(len(texts))
        np.random.shuffle(ind)
        texts = list(np.array(texts)[ind])
        labels = list(np.array(labels)[ind])

    return texts, labels


def load_trn_tst_data_labels(trn_file,
                             tst_file=None,
                             ratio=0.2,
                             split_tag='\t',
                             lbl_text_index=[0, 1],
                             is_shuffle=False):
    """
    Loads train data and test data,return segment words and labels
    if tst_file is None , split train data by ratio
    """
    # train data
    trn_data, trn_labels = load_data_and_labels(
        trn_file, split_tag, lbl_text_index, is_shuffle=is_shuffle)

    # test data
    if tst_file:
        tst_data, tst_labels = load_data_and_labels(
            tst_file, split_tag, lbl_text_index, is_shuffle=is_shuffle)
    else:
        index = np.arange(len(trn_labels))
        np.random.shuffle(index)
        split_n = int(ratio * len(trn_labels))

        trn_data = np.array(trn_data)
        trn_labels = np.array(trn_labels)

        tst_data, tst_labels = trn_data[index[:split_n]], trn_labels[
            index[:split_n]]
        trn_data, trn_labels = trn_data[index[split_n:]], trn_labels[index[
            split_n:]]

    return list(trn_data), list(trn_labels), list(tst_data), list(tst_labels)


def pad_sentences(sentences, padding_word="<PAD/>", mode='max'):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    if mode == 'max':
        sequence_length = max(len(x) for x in sentences)
    else:
        sequence_length = sum(len(x) for x in sentences) / len(sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if len(sentence) < sequence_length:
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:sequence_length]
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary, padding_word="<PAD/>"):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[
        vocabulary[word] if word in vocabulary else vocabulary[padding_word]
        for word in sentence
    ] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data(trn_file,
              tst_file=None,
              ratio=0.2,
              split_tag='\t',
              lbl_text_index=[0, 1],
              vocabulary=None,
              vocabulary_inv=None,
              padding_mod='max',
              is_shuffle=False,
              use_tst=False):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    print("%s  loading train data and label....." %
          time.asctime(time.localtime(time.time())))
    trn_text, trn_labels = load_data_and_labels(
        trn_file, split_tag, lbl_text_index, is_shuffle=is_shuffle)
    if tst_file:
        print("%s  loading test data and label....." %
              time.asctime(time.localtime(time.time())))
        tst_text, tst_labels = load_data_and_labels(
            tst_file, split_tag, lbl_text_index, is_shuffle=is_shuffle)
        sentences, labels = trn_text + tst_text, trn_labels + tst_labels
    else:
        sentences, labels = trn_text, trn_labels
    print("%s  padding sentences....." %
          time.asctime(time.localtime(time.time())))
    sentences_padded = pad_sentences(sentences, mode=padding_mod)

    if vocabulary is None or vocabulary_inv is None:
        print("%s  building vocab....." %
              time.asctime(time.localtime(time.time())))
        vocabulary, vocabulary_inv = build_vocab(sentences_padded)

    x, y = build_input_data(sentences_padded, labels, vocabulary)

    if tst_file is None and not use_tst:
        return [x, y, vocabulary, vocabulary_inv]
    elif tst_file:
        split_n = len(trn_text)
    elif use_tst:
        split_n = int((1 - ratio) * len(trn_text))

    return x[:split_n], y[:split_n], x[split_n:], y[
        split_n:], vocabulary, vocabulary_inv


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def readExcelByCol(file_name, col):
    '''
    按列读取excel数据
    :param file_name: excel路径
    :param col: 列数,可以多列,或者单列
    :return: 迭代器 可调用next(gener)来返回数据
    '''
    wb = xlrd.open_workbook(file_name)
    sheet = wb.sheet_by_index(0)

    if isinstance(col, list):
        for i in col:
            yield sheet.col_values(i)
    elif isinstance(col, int):
        data = sheet.col_values(col)
        yield data


def sample(N):
    '''
    伪造数据
    :param N:样本量
    :return:X,Y dtype:np.ndarray
    '''
    x_data, y_data = [], []
    w = np.random.normal(size=100)
    for i in range(N):
        y = np.zeros(4)
        x = np.random.random(100)
        if np.abs(np.dot(w, x)) <= 3.2:
            y[1:3] = 1
        else:
            y[0], y[-1] = 1, 1
        x = ['%.3f' % xx for xx in np.random.random(100)]
        x_data.append(x)
        y_data.append(y)

    return np.array(x_data), np.array(y_data)


def parseLable(file='../docs/CNN/dic_label'):
    '''
    解析层级标签
    :param file:
    :return:
    '''
    id_cate = {}
    G1, G2 = [], []
    with open(file, 'r') as fr:
        for line in fr:
            cont = line.decode('utf-8').split("\t")
            id_cate[int(cont[1])] = cont[0]
            if set(cont[0]) & set(['-', '_']):
                G1.append(int(cont[1]))
            else:
                G2.append(int(cont[1]))
    return id_cate, G1, G2


def loadDataSet(file):
    '''
    加载文件数据
    :param file:
    :return:
    '''
    X, Y, raw_data = [], [], []
    with open(file, 'r') as fr:
        for line in fr:
            cont = line.decode('utf-8').split('\t')
            X.append([int(x.split(":")[0]) for x in cont[1].split()])
            Y.append([int(x) for x in cont[0].split()])
            raw_data.append(cont[2])
    return X, Y, raw_data


def transY2Vec(Y, G1, G2):
    '''
    Y转化为向量
    :param Y:
    :param G1:
    :param G2:
    :return:
    '''
    G1.extend(G2)
    Y_vec = np.zeros([len(Y), len(G1)])
    for i, y in enumerate(Y):
        Y_vec[i, y] = 1

    return Y_vec


def ml_confuse(y_true, y_pre):
    '''
    calculate multi-label confuse matrix
    '''
    # init dict
    lbl_set = set(sum(y_true, []) + sum(y_pre, []))
    confuse = {}
    for lbl in lbl_set:
        confuse[lbl] = defaultdict(int)

    for i in range(len(y_true)):
        gt = set(y_true[i])
        pre = set(y_pre[i])
        if not pre:
            continue
        # predicted right labels
        pre_T = gt & pre
        # prediced wrong labels
        wrg = pre - pre_T
        if pre_T:  # hit !!
            for t_lbl in pre_T:
                confuse[t_lbl][t_lbl] += 1  # right cnt
                for w_lbl in wrg:
                    confuse[t_lbl]['+%s' % w_lbl] += 1
        # label with no predict result
        pre_w = gt - pre_T
        if pre_w:
            for t_lbl in gt - pre_T:
                for w_lbl in pre_w:
                    confuse[t_lbl]['-%s' % w_lbl] += 1

    return confuse


if __name__ == "__main__":
    import data_helpers

    print("Loading data...")
    x, _, _, vocabulary_inv = data_helpers.load_data()
    w = train_word2vec(x, vocabulary_inv)

    exit()
    id_cate, G1, G2 = parseLable()
    print(len(G1), len(G2))
    exit()
    X, Y, raw_data = loadDataSet(file='../docs/CNN/test')
    Y_vec = transY2Vec(Y, G1, G2)
