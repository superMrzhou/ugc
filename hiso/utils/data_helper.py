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
import numpy as np
import pands as pd
from collections import defaultdict
from collections import Counter
from gensim.models import word2vec
from os.path import join, exists, split


class HMultLabelSample(object):
    """ 多层标记数据样本
    @self.data: list - Feature vector of raw data
    @self.sentence_len: int - Num of words of sentence
    @self.top_label: str - The top label of data
    @self.bottom_label: str - The bottom label of data
    @self.split: int - k folds set index
    @self.vec: list - feature vector
    标记对应映射关系
    top_label_map = ['event', 'agent', 'object']
    bottom_label_map = ['Satisfaction', 'Disappointment',
        'Admiration', 'Reproach', 'Like', 'Dislike']
    """

    top_label_map = ['Event', 'Agent', 'Object']
    bottom_label_map = ['Satisfaction', 'Disappointment',
                        'Admiration', 'Reproach', 'Like', 'Dislike']

    def __init__(self, content, sentence_len, top_label, bottom_label, split):
        self.content = content
        self.sentence_len = sentence_len
        self.top_label = top_label
        self.bottom_label = bottom_label
        self.split = split
        self.vec = None

    def __str__(self):
        return ''.join([word for word, _ in self.content])

    def __len__(self):
        return self.sentence_len


def build_data_cv(file_path, cv=5):
    """ 从文件载入数据， 每个样本是一个HMultLabelSample对象
    @file_path: data file path
    @cv: k folds set
    @rtype: list(HMultLabelSample)
    """
    pd_data = pd.read_pickle(file_path)
    rev = []
    vocab = defaultdict(int)
    for i in range(pd_data.shape[0]):
        content = pd_data['Cut'][i]
        sentence_len = pd_data['Len'][i]
        top_label = [pd_data['Event'][i], pd_data['Agent'][i], pd_data['Object'][i]]
        bottom_label = [pd_data['Satisfaction'][i], pd_data['Disappointment'][i], pd_data['Admiration'][i],
                        pd_data['Reproach'][i], pd_data['Like'][i], pd_data['Dislike'][i]]

        split = np.random.randint(0, cv)
        datum = HMultLabelSample(
            content, sentence_len, top_label, bottom_label, split)
        rev.append(datum)

        words = set([word for word, _ in content])
        for word in words:
            vocab[word] += 1
    return rev, vocab


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
        sentences = [[vocabulary_inv[w] for w in filter(
            lambda w_id: w_id > 0, s)] for s in sentence_matrix]
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
    string = re.sub(r"[^A-Za-z(),!?！？，。；’‘“”’\'\`]", " ", string)

    return string.strip().lower()


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


def process_line(texts,
                 labels,
                 vocabulary,
                 category,
                 cate_split_n,
                 use_G1=True,
                 sequence_length=256,
                 padding_word='<PAD/>'):
    """Process line data to train format."""

    # padding
    if len(texts) < sequence_length:
        texts = texts + [padding_word] * (sequence_length - len(texts))
    else:
        texts = texts[:sequence_length]
    # build_input_data
    texts = [
        vocabulary[word] if word in vocabulary else vocabulary[padding_word]
        for word in texts
    ]

    # labels to vecs
    if use_G1:
        labels = list(
            set(['%s_G1' % re.split('-|_', lbl)[0] for lbl in labels] + labels))
    else:
        labels = list(
            set([re.split('-|_', lbl)[0] for lbl in labels] + labels))
    labels_vec = np.zeros(len(category))
    labels_vec[[category.index(lbl) for lbl in labels]] = 1

    return texts, labels_vec[:cate_split_n], labels_vec[cate_split_n:]


def ml_confuse(y_true, y_pre):
    '''
    calculate multi-label confuse matrix
    '''
    # init dict
    lbl_set = set(sum(y_true, []) + sum(y_pre, []))
    confuse = {}
    for lbl in lbl_set:
        confuse[lbl] = defaultdict(int)
    N = len(y_true)
    for i in range(N):
        gt = set(y_true[i])
        pre = set(y_pre[i])
        if not pre:
            continue
        # predicted right labels
        pre_T = gt & pre
        # prediced wrong labels
        wrg = pre - pre_T
        for t_lbl in pre_T:
            confuse[t_lbl][t_lbl] += 1  # right cnt
            for w_lbl in wrg:
                confuse[t_lbl]['+%s' % w_lbl] += 1
        # label with no predict result
        pre_w = gt - pre_T
        for t_lbl in pre_w:
            for w_lbl in wrg:
                confuse[t_lbl]['-%s' % w_lbl] += 1
        if i % int(0.1 * N) == 0:
            print('%s / %s' % (i, N))

    return confuse


if __name__ == "__main__":
    pass
