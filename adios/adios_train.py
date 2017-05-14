# -*- coding:utf-8 -*-
import os, re, sys
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

def train(train_dataset,valid_dataset,test_dataset,params):


    # Assemble and compile the model
    model = assemble('ADIOS', params)

    # Prepare embedding layer weights and convert inputs for static model
    model_type = params['iter']['model_type']
    print("Model type is", model_type)
    if model_type == "CNN-non-static" or model_type == "CNN-static":
        embedding_weights = train_word2vec(np.vstack((train_dataset['X'], valid_dataset['X'],test_dataset['X'])), vocabulary_inv, num_features=embedding_dim,
                                       min_word_count=min_word_count, context=context)

        train_dataset['X'] = embedding_weights[0][train_dataset['X']]
        test_dataset['X'] = embedding_weights[0][test_dataset['X']]
        valid_dataset['X'] = embedding_weights[0][valid_dataset['X']]

        if params['iter']['model_type'] == "CNN-non-static":
            embedding_layer = model.get_layer('embedding')
            embedding_layer.set_weights(embedding_weights)
    elif model_type == "CNN-rand":
        embedding_weights = None
    else:
        raise ValueError("Unknown model type")

    # complie model
    model.compile(loss={'Y0': 'binary_crossentropy',
                        'Y1': 'binary_crossentropy'},
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
              verbose=1)

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


def y2vec(y,cate_id,cateIds_list):


    res = np.zeros((len(y),len(cateIds_list)))
    for i,yy in enumerate(y):
        res[i][[cateIds_list.index(cate_id[lbl]) for lbl in yy]] = 1
    return res

def y2list(y):

    y = [yy[0].strip('\n').split('&') for yy in y]
    return [list(set([re.split('-|_',lbl)[0] for lbl in yy])) + yy for yy in y]


def get_Y0_and_Y1(file_path):
    with open(file_path,'r') as f_cate:
        Y0,Y1 = [],[]
        for line in f_cate:
            if re.search('-|_',line.split('\t')[0]):
                Y0.append(line.strip('\n').split('\t')[1])
            else:
                Y1.append(line.strip('\n').split('\t')[1])
    return Y0,Y1

def filter_data(x,y):
    res_x,res_y = [],[]
    for i,yy in enumerate(y):
        temp_y = filter(lambda lbl: '其他' not in lbl and '新闻' not in lbl,yy)
        if temp_y:
            res_x.append(x[i])
            res_y.append(temp_y)
    return res_x,res_y


if __name__ == '__main__':

    vocabulary_inv,_ = load_data_and_labels('../docs/CNN/dic_v5',lbl_text_index=[1,0])
    vocabulary_inv = [x[0] for x in vocabulary_inv]
    vocabulary_inv.insert(0,'<PAD/>')
    vocabulary = {x: i for i,x in enumerate(vocabulary_inv)}

    # Load the datasets
    trn_text,trn_labels,tst_text,tst_labels,vocabulary,vocabulary_inv = load_data('../docs/CNN/split_ab',
                                                                                    use_tst=True,
                                                                                    lbl_text_index=[1,3],
                                                                                    split_tag='@@@',
                                                                                    ratio=0.2,
                                                                                    vocabulary=vocabulary,
                                                                                    vocabulary_inv=vocabulary_inv)

    Y1,Y0 = get_Y0_and_Y1('../docs/CNN/cate_id')
    print('Y0 size : %d , Y1 size : %d'%(len(Y0),len(Y1)))
    cates,ids = load_data_and_labels('../docs/CNN/cate_id',lbl_text_index=[1,0])

    cate_id = dict(zip([cate[0] for cate in cates],[_id[0] for _id in ids]))

    # add first cate
    trn_labels = y2list(trn_labels)
    tst_labels = y2list(tst_labels)

    # filter 其他 and 新闻
    trn_text,trn_labels = filter_data(trn_text,trn_labels)
    tst_text,tst_labels = filter_data(tst_text,tst_labels)

    # vectorize
    trn_labels = y2vec(trn_labels,cate_id,Y0+Y1)
    tst_labels = y2vec(tst_labels,cate_id,Y0+Y1)

    # params
    nb_features = len(vocabulary_inv)
    nb_labels = len(Y0 + Y1)
    nb_labels_Y0 = len(Y0)
    nb_labels_Y1 = len(Y1)

    trn_text = np.array(trn_text)
    tst_text = np.array(tst_text)
    print('train data size : %d , test data size : %d'%(len(trn_labels),len(tst_labels)))
    print('X sequence_length is : %d , Y dim : %d'%(trn_text.shape[1],trn_labels.shape[1]))
    # load params config
    params = yaml.load(open('../docs/configs/adios.yaml'))
    params['X']['sequence_length'] = trn_text.shape[1]
    params['X']['vocab_size'] = len(vocabulary)
    params['Y0']['dim'] = nb_labels_Y0
    params['Y1']['dim'] = nb_labels_Y1
    print(params)
    # Specify datasets in the format of dictionaries
    ratio = 0.2
    valid_N = int(ratio * tst_text.shape[0])
    train_dataset = {'X': trn_text,
                     'Y0': trn_labels[:,:nb_labels_Y0],
                     'Y1': trn_labels[:,nb_labels_Y0:]}
    valid_dataset = {'X': tst_text[:valid_N],
                     'Y0': tst_labels[:valid_N,:nb_labels_Y0],
                     'Y1': tst_labels[:valid_N,nb_labels_Y0:]}
    test_dataset = {'X': tst_text[valid_N:],
                    'Y0': tst_labels[valid_N:,:nb_labels_Y0],
                    'Y1': tst_labels[valid_N:,nb_labels_Y0:]}

    # start train
    train(train_dataset,valid_dataset,test_dataset,params)
