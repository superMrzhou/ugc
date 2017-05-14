# -*- coding:utf-8 -*-
import os, re, os
reload(sys)
sys.setdefaultencoding('utf8')

import yaml
import numpy as np
from copy import deepcopy

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adagrad

from adios.utils.callbacks import HammingLoss
from adios.utils.metrics import f1_measure, hamming_loss, precision_at_k
from adios.utils.assemble import assemble
from adios.utils.data_helper import *

def train(train_dataset,valid_dataset,test_dataset,params):


    # Assemble and compile the model
    model = assemble('ADIOS', params)
    model.compile(loss={'Y0': 'binary_crossentropy',
                        'Y1': 'binary_crossentropy'},
                  optimizer=Adagrad(1e-1))

    embedding_layer = model.get_layer('embedding')
    embedding_layer.set_weights()
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
        # EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='auto'),
    ] # TODO 早停止参数需要进一步确定 (zhangliujie)

    # Fit the model to the data
    batch_size = params['iter']['batch_size']
    nb_epoch = params['iter']['epoch']

    # start to train
    model.fit(x=train_dataset["X"],
              y=[train_dataset['Y0'], train_dataset['Y1']],
              validation_data=(valid_dataset["X"],[valid_dataset["Y0"],valid_dataset["Y1"]]),
              batch_size=batch_size,
              epochs=nb_epoch,
              callbacks=callbacks,
              verbose=2)

    # Load the best weights
    if os.path.isfile(model_dir + model_name):
        model.load_weights(model_dir + model_name)

    # Fit thresholds
    model.fit_thresholds(train_dataset, validation_data=valid_dataset,
                         alpha=np.logspace(-3, 3, num=10).tolist(), verbose=1)

    # Test the model
    probs, preds = model.predict_threshold(test_dataset)

    hl = hamming_loss(test_dataset, preds)
    f1_macro = f1_measure(test_dataset, preds, average='macro')
    f1_micro = f1_measure(test_dataset, preds, average='micro')
    f1_samples = f1_measure(test_dataset, preds, average='samples')
    p_at_1 = precision_at_k(test_dataset, probs, K=1)
    # p_at_5 = precision_at_k(test_dataset, probs, K=5)
    # p_at_10 = precision_at_k(test_dataset, probs, K=10)

    for k in ['Y0', 'Y1', 'all']:
        print
        print("Hamming loss (%s): %.2f" % (k, hl[k]))
        print("F1 macro (%s): %.4f" % (k, f1_macro[k]))
        print("F1 micro (%s): %.4f" % (k, f1_micro[k]))
        print("F1 sample (%s): %.4f" % (k, f1_samples[k]))
        print("P@1 (%s): %.4f" % (k, p_at_1[k]))
        # print("P@5 (%s): %.4f" % (k, p_at_5[k]))
        # print("P@10 (%s): %.4f" % (k, p_at_10[k]))
def prepare():
    X,Y,cate_id = [],[],{}
    _id = 0
    line_n = 0
    with open('../docs/CNN/toutiao_category_video_v5','r') as f:
        for line in f:
            cont = line.decode('utf-8').split("@@@")
            labels = cont[1].strip().split('&')
            word_ids = cont[2].strip().split()
            if len(labels) == 1 and ('其他' in labels[0] or '新闻' in labels[0]):
                continue
            for label in labels:
                if '其他' in label or '新闻' in label:
                    continue
                cates = set()
                if label not in cate_id:
                    cate_id[label] = _id
                    _id += 1
                head = re.split('-|_',label)[0]
                if head not in cate_id:
                    cate_id[head] = _id
                    _id += 1
                cates.add(str(cate_id[label]))
                cates.add(str(cate_id[head]))
            Y.append(list(cates))
            X.append(word_ids)
            line_n += 1
            if line_n % 10000 == 0:
                print(line_n)
    # split train and test
    ratio = 0.8
    X = np.array(X)
    Y = np.array(Y)
    indexs = np.arange(len(X))
    np.random.shuffle(indexs)
    split_n = int(ratio * len(X))
    with open('../docs/CNN/train','w') as f_trn, open('../docs/CNN/test','w') as f_tst:
        for i in range(len(X)):
            cont = ' '.join(Y[indexs[i]]) + '\t' + ' '.join(X[indexs[i]]) + '\n'
            if i < split_n:
                f_trn.write(cont)
            else:
                f_tst.write(cont)
    with(open('../docs/CNN/cate_id','w')) as f:
        for cate,id in cate_id.items():
            f.write('%s\t%s\n'%(cate,id))

def loadData():
    with open('../docs/CNN/train','r') as f_trn,\
         open('../docs/CNN/test','r') as f_tst,\
         open('../docs/CNN/cate_id','r') as f_cate:
        Y0,Y1 = [],[]
        for line in f_cate:
            if re.search('-|_',line.split('\t')[0]):
                Y0.append(line.strip('\n').split('\t')[1])
            else:
                Y1.append(line.strip('\n').split('\t')[1])

        X_train,Y_train = [],[]
        for line in f_trn:
            X_train.append(list(map(lambda x:int(x),line.strip('\n').split('\t')[1].split())))
            Y_train.append(line.split('\t')[0].split())

        X_test,Y_test = [],[]
        for line in f_tst:
            X_test.append(list(map(lambda x:int(x),line.strip('\n').split('\t')[1].split())))
            Y_test.append(line.split('\t')[0].split())

    return X_train,Y_train,X_test,Y_test,Y0,Y1

def y2vec(y,Y0,Y1):
    Y = deepcopy(Y0)
    Y.extend(Y1)
    res = np.zeros((len(y),len(Y)))
    for i,yy in enumerate(y):
        res[i][list(map(lambda lbl: Y.index(lbl), yy))] = 1
    return res

def x2vec(X_train,X_test):

    dic_szie = max(max(X_train) + max(X_test)) + 1

    train_vec,test_vec = [],[]
    for i in range(len(X_train)):
        norm_vec = np.zeros(dic_szie)
        norm_vec[X_train[i]] = 1
        train_vec.append(norm_vec)

    for i in range(len(X_test)):
        norm_vec = np.zeros(dic_szie)
        norm_vec[X_test[i]] = 1
        test_vec.append(norm_vec)
    return np.array(train_vec),np.array(test_vec)


if __name__ == '__main__':

    # Load the datasets
    X_train,Y_train,X_test,Y_test,Y0,Y1 = loadData()
    max_len = max(map(len,X_train))

    dic_szie = max(max(X_train) + max(X_test)) + 1
    print(dic_szie)
    # padding
    X_train = sequence.pad_sequences(X_train,
                                     maxlen=max_len,
                                     padding='post',
                                     truncating='post')[:100]

    X_test = sequence.pad_sequences(X_test,
                                     maxlen=max_len,
                                     padding='post',
                                     truncating='post')[:100]
    # vectorize
    Y_train = y2vec(Y_train,Y0,Y1)[:100]
    Y_test = y2vec(Y_test,Y0,Y1)[:100]

    # params
    nb_features = dic_szie
    nb_labels = Y_train.shape[1]
    nb_labels_Y0 = len(Y0)
    nb_labels_Y1 = len(Y1)

    # load params config
    params = yaml.load(open('../docs/configs/adios.yaml'))
    params['X']['dim'] = max_len
    params['X']['vocab_size'] = dic_szie
    params['Y0']['dim'] = nb_labels_Y0
    params['Y1']['dim'] = nb_labels_Y1
    print(params)
    # Specify datasets in the format of dictionaries
    ratio = 0.2
    valid_N = int(ratio * X_test.shape[0])
    train_dataset = {'X': X_train,
                     'Y0': Y_train[:,:nb_labels_Y0],
                     'Y1': Y_train[:,nb_labels_Y0:]}
    valid_dataset = {'X': X_test[:valid_N],
                     'Y0': Y_test[:valid_N,:nb_labels_Y0],
                     'Y1': Y_test[:valid_N,nb_labels_Y0:]}
    test_dataset = {'X': X_test[valid_N:],
                    'Y0': Y_test[valid_N:,:nb_labels_Y0],
                    'Y1': Y_test[valid_N:,nb_labels_Y0:]}

    # start train
    train(train_dataset,valid_dataset,test_dataset,params)
