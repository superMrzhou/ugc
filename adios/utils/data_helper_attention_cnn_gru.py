# -*- coding:utf-8 -*-
"""
@version: 1.0
@author: kevin
@license: Apache Licence
@contact: liujiezhang@bupt.edu.cn
@site:
@software: Atom
@file: data_helper_attention_cnn_gru.py
@time: 17/06/30 14:31
"""
import itertools
import os
import re
import sys
import time
from collections import Counter, defaultdict
from math import ceil
from os.path import exists, join, split

import numpy as np
from gensim.models import word2vec

reload(sys)
sys.setdefaultencoding('utf8')


def load_feature_title_content(train_path, feature_title_path,
                               feature_content_path):
    with open(train_path, 'r') as f:
        # title feature
        vocabulary_set = set()
        for data in f:
            for word in data.strip('\n').split("@@@")[1].strip(" ").split(" "):
                vocabulary_set.add(word)
        vocabulary_inv = list(vocabulary_set)
        padding_string = '<PAD/>'
        vocabulary_inv.insert(0, padding_string)
        output = open(feature_title_path, 'a')
        for i, v in enumerate(vocabulary_inv):
            output.write(str(v) + "\t" + str(i) + "\n")
        output.close()
    with open(train_path, 'r') as f:
        # title feature
        vocabulary_set = set()
        for data in f:
            for word in data.strip('\n').split("@@@")[2].strip(" ").split(" "):
                vocabulary_set.add(word)
        vocabulary_inv = list(vocabulary_set)
        padding_string = '<PAD/>'
        vocabulary_inv.insert(0, padding_string)
        output = open(feature_content_path, 'a')
        for i, v in enumerate(vocabulary_inv):
            output.write(str(v) + "\t" + str(i) + "\n")
        output.close()


def load_labels_title_content(file_path,
                              split_tag='\t',
                              lbl_text_index=[0, 1, 2],
                              is_shuffle=True):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """

    # Load data from files
    with open(file_path, 'r') as f:
        # parse label
        labels = [
            data.strip('\n').split(split_tag)[lbl_text_index[0]] for data in f
        ]
    with open(file_path, 'r') as f:
        # parse text
        titles = [
            data.strip('\n').split(split_tag)[lbl_text_index[1]] for data in f
        ]
    with open(file_path, 'r') as f:
        # parse text
        contents = [
            data.strip('\n').split(split_tag)[lbl_text_index[2]] for data in f
        ]

    # Split by words
    contents = [filter(lambda a: a != '', s.split(" ")) for s in contents]
    titles = [filter(lambda a: a != '', s.split(" ")) for s in titles]
    # support multi-label
    labels = [filter(lambda a: a != '', s.split(" ")) for s in labels]
    if is_shuffle:
        ind = np.arange(len(titles))
        np.random.shuffle(ind)
        titles = np.array(titles)[ind].tolist()
        contents = np.array(contents)[ind].tolist()
        labels = np.array(labels)[ind].tolist()

    return labels, titles, contents


def train_word2vec_params(params, vocabulary_title_inv,
                          vocabulary_content_inv):
    """
      Trains, saves, loads Word2Vec model
      Returns initial weights for embedding layer.

      inputs:
      file_name=>sentence_matrix # int matrix: num_sentences x max_sentence_len
      vocabulary_inv  # list
      num_features    # Word vector dimensionality
      min_word_count  # Minimum word count
      context         # Context window size
      """
    model_dir = os.path.abspath('../docs') + '/model_new/w2v_matrix'
    model_title = "{:d}features_{:d}minwords_{:d}context_title".format(
        len(vocabulary_title_inv), 1, 5)
    model_content = "{:d}features_{:d}minwords_{:d}context_content".format(
        len(vocabulary_content_inv), 1, 5)
    model_title = join(model_dir, model_title)
    model_content = join(model_dir, model_content)
    if exists(model_title) and exists(model_content):
        embedding_model_title = word2vec.Word2Vec.load(model_title)
        embedding_model_content = word2vec.Word2Vec.load(model_content)
        print('Load existing Word2Vec model \'%s\'' % split(model_title)[-1])
        print('Load existing Word2Vec model \'%s\'' % split(model_content)[-1])
    else:
        # Set values for various parameters
        num_workers = 10  # Number of threads to run in parallel
        downsampling = 1e-3  # Downsample setting for frequent words

        # Initialize and train the model
        print('Training Word2Vec model...')
        # labels, titles, contents = load_labels_title_content(
        #     file_name, "@@@", [0, 1, 2], is_shuffle=True)

        with open(params['train_path'], 'r') as f:
            titles = [x.strip('\n').split("@@@")[1].split(" ") for x in f]

        embedding_model_title = word2vec.Word2Vec(
            titles,
            workers=num_workers,
            size=params['title_layer']['embedding_dic_dim'],
            min_count=1,
            window=5,
            sample=downsampling)
        # If we don't plan to train the model any further, calling
        # init_sims will make the model much more memory-efficient.
        embedding_model_title.init_sims(replace=True)
        # Saving the model for later use. You can load it later using
        # Word2Vec.load()
        if not exists(model_dir):
            os.mkdir(model_dir)
        print('Saving Word2Vec model \'%s\'' % split(model_title)[-1])
        embedding_model_title.save(model_title)

        with open(params['train_path'], 'r') as f:
            contents = [x.strip('\n').split("@@@")[2].split(" ") for x in f]
        embedding_model_content = word2vec.Word2Vec(
            contents,
            workers=num_workers,
            size=params['title_layer']['embedding_dic_dim'],
            min_count=1,
            window=5,
            sample=downsampling)
        # If we don't plan to train the model any further, calling
        # init_sims will make the model much more memory-efficient.
        embedding_model_content.init_sims(replace=True)
        # Saving the model for later use. You can load it later using
        # Word2Vec.load()
        print('Saving Word2Vec model \'%s\'' % split(model_content)[-1])
        embedding_model_content.save(model_content)

    # add unknown words
    title_embedding_weights = [
        np.array([
            embedding_model_title[w]
            if w in embedding_model_title else np.random.uniform(
                -0.25, 0.25, embedding_model_title.vector_size)
            for w in vocabulary_title_inv
        ])
    ]
    content_embedding_weights = [
        np.array([
            embedding_model_content[w]
            if w in embedding_model_content else np.random.uniform(
                -0.25, 0.25, embedding_model_content.vector_size)
            for w in vocabulary_content_inv
        ])
    ]
    return title_embedding_weights, content_embedding_weights


def train_word2vec(file_name,
                   vocabulary_inv,
                   num_features=100,
                   min_word_count=2,
                   context=5):
    """
    Trains, saves, loads Word2Vec model
    Returns initial weights for embedding layer.

    inputs:
    file_name=>sentence_matrix # int matrix: num_sentences x max_sentence_len
    vocabulary_inv  # list
    num_features    # Word vector dimensionality
    min_word_count  # Minimum word count
    context         # Context window size
    """
    model_dir = os.path.abspath('../docs') + '/model_new/w2v_matrix'
    model_title = "{:d}features_{:d}minwords_{:d}context_title".format(
        num_features, min_word_count, context)
    model_content = "{:d}features_{:d}minwords_{:d}context_content".format(
        num_features, min_word_count, context)
    model_title = join(model_dir, model_title)
    model_content = join(model_dir, model_content)
    if exists(model_title) and exists(model_content):
        embedding_model_title = word2vec.Word2Vec.load(model_title)
        embedding_model_content = word2vec.Word2Vec.load(model_content)
        print('Load existing Word2Vec model \'%s\'' % split(model_title)[-1])
        print('Load existing Word2Vec model \'%s\'' % split(model_content)[-1])
    else:
        # Set values for various parameters
        num_workers = 10  # Number of threads to run in parallel
        downsampling = 1e-3  # Downsample setting for frequent words

        # Initialize and train the model
        print('Training Word2Vec model...')
        labels, titles, contents = load_labels_title_content(
            file_name, "@@@", [0, 1, 2], is_shuffle=True)
        embedding_model_title = word2vec.Word2Vec(
            titles,
            workers=num_workers,
            size=num_features,
            min_count=min_word_count,
            window=context,
            sample=downsampling)
        # If we don't plan to train the model any further, calling
        # init_sims will make the model much more memory-efficient.
        embedding_model_title.init_sims(replace=True)
        # Saving the model for later use. You can load it later using
        # Word2Vec.load()
        if not exists(model_dir):
            os.mkdir(model_dir)
        print('Saving Word2Vec model \'%s\'' % split(model_title)[-1])
        embedding_model_title.save(model_title)

        embedding_model_content = word2vec.Word2Vec(
            contents,
            workers=num_workers,
            size=num_features,
            min_count=min_word_count,
            window=context,
            sample=downsampling)
        # If we don't plan to train the model any further, calling
        # init_sims will make the model much more memory-efficient.
        embedding_model_content.init_sims(replace=True)
        # Saving the model for later use. You can load it later using
        # Word2Vec.load()
        print('Saving Word2Vec model \'%s\'' % split(model_content)[-1])
        embedding_model_content.save(model_content)

    # add unknown words
    title_embedding_weights = [
        np.array([
            embedding_model_title[w]
            if w in embedding_model_title else np.random.uniform(
                -0.25, 0.25, embedding_model_title.vector_size)
            for w in vocabulary_inv
        ])
    ]
    content_embedding_weights = [
        np.array([
            embedding_model_content[w]
            if w in embedding_model_content else np.random.uniform(
                -0.25, 0.25, embedding_model_content.vector_size)
            for w in vocabulary_inv
        ])
    ]
    return title_embedding_weights, content_embedding_weights


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
                         is_shuffle=True):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """

    # Load data from files
    with open(file_path, 'r') as f:
        # parse label
        labels = [
            data.strip('\n').split(split_tag)[lbl_text_index[0]] for data in f
        ]
    with open(file_path, 'r') as f:
        # parse text
        texts = [
            data.strip('\n').split(split_tag)[lbl_text_index[1]] for data in f
        ]

    # Split by words
    # texts = [clean_str(sent) for sent in texts]
    texts = [filter(lambda a: a != '', s.split(" ")) for s in texts]
    # support multi-label
    labels = [filter(lambda a: a != '', s.split(" ")) for s in labels]
    if is_shuffle:
        ind = np.arange(len(texts))
        np.random.shuffle(ind)
        texts = np.array(texts)[ind].tolist()
        labels = np.array(labels)[ind].tolist()

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
              is_shuffle=True,
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


def process_line(title,
                 content,
                 labels,
                 vocabulary,
                 category,
                 cate_split_n,
                 use_G1=True,
                 title_sequence_length=30,
                 content_sequence_length=256,
                 padding_word='<PAD/>'):
    """Process line data to train format."""

    # padding title
    if len(title) < title_sequence_length:
        title = title + [padding_word] * (title_sequence_length - len(title))
    else:
        title = title[:title_sequence_length]
    # build_input_data
    title = [
        vocabulary[word] if word in vocabulary else vocabulary[padding_word]
        for word in title
    ]
    # padding content
    if len(content) < content_sequence_length:
        content = content + [padding_word] * (
            content_sequence_length - len(content))
    else:
        content = content[:content_sequence_length]
    # build_input_data
    content = [
        vocabulary[word] if word in vocabulary else vocabulary[padding_word]
        for word in content
    ]

    # labels to vecs
    if use_G1:
        labels = list(
            set(['%s_G1' % re.split('-|_', lbl)[0]
                 for lbl in labels] + labels))
    else:
        labels = list(
            set([re.split('-|_', lbl)[0] for lbl in labels] + labels))
    labels_vec = np.zeros(len(category))
    labels_vec[[category.index(lbl) for lbl in labels]] = 1

    return title, content, labels_vec[:cate_split_n], labels_vec[cate_split_n:]


def generate_arrays_from_dataset(file_path,
                                 vocabulary,
                                 category,
                                 cate_split_n,
                                 split_tag='\t',
                                 lbl_text_index=[0, 1, 2],
                                 batch_size=2048,
                                 shuffle_batch_num=100,
                                 title_sequence_length=30,
                                 content_sequence_length=256,
                                 use_G1=True,
                                 padding_word='<PAD/>'):
    """Data generator."""
    while 1:
        line_cnt = 0
        titles, contents, labels = [], [], []
        with open(file_path, 'r') as f:
            for line in f:
                if line and line_cnt < batch_size * shuffle_batch_num:
                    titles.append(
                        line.strip('\n').split(split_tag)[lbl_text_index[1]]
                        .split())
                    contents.append(
                        line.strip('\n').split(split_tag)[lbl_text_index[2]]
                        .split())
                    labels.append(
                        line.strip('\n').split(split_tag)[lbl_text_index[0]]
                        .split())
                    line_cnt += 1
                    continue
                # shuffle
                local_num = len(titles)
                ind = np.arange(local_num)
                np.random.shuffle(ind)
                titles = np.array(titles)[ind]
                contents = np.array(contents)[ind]
                labels = np.array(labels)[ind]

                batch_num = int(ceil(local_num / float(batch_size)))
                for current_batch in range(batch_num):
                    start = current_batch * batch_size
                    step = min(batch_size, local_num - start)
                    batch_titles = titles[start:start + step]
                    batch_contents = contents[start:start + step]
                    batch_labels = labels[start:start + step]
                    res = [
                        process_line(
                            batch_titles[i],
                            batch_contents[i],
                            batch_labels[i],
                            vocabulary,
                            category,
                            cate_split_n,
                            use_G1=True,
                            title_sequence_length=title_sequence_length,
                            content_sequence_length=content_sequence_length,
                            padding_word=padding_word)
                        for i in range(len(batch_labels))
                    ]
                    res = zip(*res)
                    yield ({
                        'title': np.array(res[0]),
                        'content': np.array(res[1])
                    }, {
                        'Y0': np.array(res[2]),
                        'Y1': np.array(res[3])
                    })
                titles, contents, labels = [], [], []


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
