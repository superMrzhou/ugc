# -*- coding:utf-8 -*-
"""
@version: 1.0
@author: kevin
@license: Apache Licence
@contact: liujiezhang@bupt.edu.cn
@site:
@software: Atom
@file: gru_cnn_model.py
@time: 17/06/30 11:39
"""
import gc

import numpy as np
from keras import backend as K
from keras import initializers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine.topology import Layer
from keras.layers import (GRU, Activation, ActivityRegularization,
                          BatchNormalization, Conv1D, Dense, Dropout,
                          Embedding, Flatten, Input, Lambda, MaxPooling1D,
                          Reshape, concatenate, merge)
from keras.layers.core import Permute, RepeatVector
from keras.layers.wrappers import Bidirectional
from keras.metrics import categorical_accuracy
from keras.optimizers import RMSprop
from keras.regularizers import l2
from utils.callbacks import HammingLoss
from utils.data_helper_attention_cnn_gru import (generate_arrays_from_dataset,
                                                 train_word2vec)
from utils.metrics import f1_measure, precision_at_k
from utils.models import MLC_GRU


class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        # self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1], ))
        # self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer,
              self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        weights = ai / K.sum(ai, axis=1).dimshuffle(0, 'x')

        weighted_input = x * weights.dimshuffle(0, 1, 'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])


def attention_3d_block(inputs,
                       activation='softmax',
                       single_attention_vector=False):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    TIME_STEPS = int(inputs.shape[1])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a)
    a = Dense(TIME_STEPS, activation=activation)(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge(
        [inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul


class GRU_CNN_Attention():
    '''Build and Train model while compile title by GRU and content by CNN.'''

    def __init__(self, params):
        self.params = params
        self.build()

    def build(self):
        # multi input from title and content
        content_sequence_input = Input(
            shape=(self.params['content_layer']['sequence_length'], ),
            name='content',
            dtype='int32')
        content_embedding_layer = Embedding(
            input_dim=self.params['content_layer'][
                'embedding_dic_dim'],  # dic dim
            output_dim=self.params['content_layer'][
                'embedding_dim'],  # embedding dim
            # sequence length
            input_length=self.params['content_layer']['sequence_length'],
            name='content_embedding_layer',
            trainable=True)(content_sequence_input)

        title_sequence_input = Input(
            shape=(self.params['title_layer']['sequence_length'], ),
            name='title',
            dtype='int32')
        title_embedding_layer = Embedding(
            input_dim=self.params['title_layer'][
                'embedding_dic_dim'],  # dic dim
            output_dim=self.params['title_layer'][
                'embedding_dim'],  # embedding dim
            input_length=self.params['title_layer'][
                'sequence_length'],  # sequence length
            name='title_embedding_layer',
            trainable=True,
            mask_zero=True)(title_sequence_input)

        # Bidirectional title_hidden
        # left_gru = GRU(
        #     self.params['title_layer']['gru_cell'])(title_embedding_layer)
        # right_gru = GRU(
        #     self.params['title_layer']['gru_cell'], go_backwards=True)(title_embedding_layer)
        # conebime_gru = Merge([left_gru, right_gru], mode='sum')
        title_hidden = Bidirectional(
            GRU(self.params['title_layer']['gru_cell'],
                return_sequences=True))(title_embedding_layer)
        # attention machanism
        # title_hidden = AttLayer(bi_gru)
        title_hidden = BatchNormalization(
            name='title_batchNorm', momentum=0.9)(title_hidden)
        Y0_input = BatchNormalization(momentum=0.9)(
            GRU(self.params['Y0']['dim'], name='Y0_input')(title_hidden))
        Y0 = Activation('softmax')(Y0_input)
        Y0 = ActivityRegularization(
            name='Y0', **self.params['Y0']['activity_reg'])(Y0)

        # content_hidden
        content_hidden = Conv1D(
            128, 3, activation='relu')(content_embedding_layer)
        content_hidden = MaxPooling1D(3)(content_hidden)
        content_hidden = BatchNormalization(
            name='content_batchNorm1', momentum=0.9)(content_hidden)
        content_hidden = Dropout(0.3)(content_hidden)

        content_hidden = Conv1D(128, 4, activation='relu')(content_hidden)
        content_hidden = MaxPooling1D(4)(content_hidden)
        content_hidden = BatchNormalization(
            name='content_batchNorm2', momentum=0.9)(content_hidden)
        content_hidden = Dropout(0.3)(content_hidden)

        content_hidden = Conv1D(128, 5, activation='relu')(content_hidden)
        content_hidden = MaxPooling1D(5)(content_hidden)  # global max pooling
        content_hidden = BatchNormalization(
            name='content_batchNorm3', momentum=0.9)(content_hidden)
        content_hidden = Dropout(0.3)(content_hidden)
        # Flatten feature
        content_hidden = Flatten()(content_hidden)

        # combine feature
        Y0_H0 = concatenate([Y0_input, content_hidden])

        # add hidden between Y1 and combine-feature layer
        H1 = Dense(
            self.params['H1']['dim'],
            activation='relu',
            bias_regularizer=l2(0.01))(Y0_H0)
        H1 = BatchNormalization(
            name='H1_batch_norm', **self.params['H1']['batch_norm'])(H1)
        H1 = Dropout(0.3, name='H')(H1)

        # predict Y1
        Y1 = Dense(
            self.params['Y1']['dim'],
            activation='softmax',
            name='Y1_activation',
            bias_regularizer=l2(0.01))(H1)
        Y1 = ActivityRegularization(
            name='Y1', **self.params['Y1']['activity_reg'])(Y1)
        self.model = MLC_GRU(
            inputs=[title_sequence_input, content_sequence_input],
            outputs=[Y0, Y1])
        return self.model

    def train(self,
              train_data_file,
              w2v_preTrain_file,
              vocabulary_inv,
              category,
              valid_data=None,
              test_data_file=None):
        '''Train model.'''
        vocabulary = {k: v for v, k in enumerate(vocabulary_inv)}
        # train_word2vec automally
        title_embedding_weights, content_embedding_weights = train_word2vec(
            file_name=w2v_preTrain_file,
            vocabulary_inv=vocabulary_inv,
            num_features=self.params['title_layer']['embedding_dim'],
            min_word_count=1,
            context=5)
        # init embedding layer weights
        content_embedding_layer = self.model.get_layer(
            'content_embedding_layer')
        content_embedding_layer.set_weights(content_embedding_weights)

        title_embedding_layer = self.model.get_layer('title_embedding_layer')
        title_embedding_layer.set_weights(title_embedding_weights)

        self.model.compile(
            loss={
                'Y0': self.params['Y0']['loss_func'],
                'Y1': self.params['Y1']['loss_func']
            },
            loss_weights={
                'Y0': self.params['Y0']['loss_weight'],
                'Y1': self.params['Y1']['loss_weight']
            },
            metrics=[categorical_accuracy],
            optimizer=RMSprop(lr=self.params['iter']['learn_rate']))
        model_dir = self.params['iter']['model_dir']
        model_name = self.params['iter']['model_name']
        callbacks = [HammingLoss({'valid': valid_data})] if valid_data else []

        callbacks.extend([
            ModelCheckpoint(
                model_dir + model_name,
                monitor='val_hl',
                verbose=0,
                save_best_only=True,
                mode='min'),
            EarlyStopping(
                monitor='val_loss', patience=10, verbose=1, mode='min')
        ])
        data_generator = generate_arrays_from_dataset(
            train_data_file,
            vocabulary,
            category,
            self.params['Y0']['dim'],
            split_tag='@@@',
            lbl_text_index=[0, 1, 2],
            batch_size=self.params['iter']['batch_size'],
            title_sequence_length=self.params['title_layer'][
                'sequence_length'],
            content_sequence_length=self.params['content_layer'][
                'sequence_length'],
            use_G1=True,
            padding_word='<PAD/>')
        self.model.fit_generator(
            data_generator,
            validation_data=({
                'title': valid_data['title'],
                'content': valid_data['content']
            }, {
                'Y0': valid_data['Y0'],
                'Y1': valid_data['Y1']
            }),
            steps_per_epoch=self.params['iter']['steps_per_epoch'],
            epochs=self.params['iter']['epoch'],
            callbacks=callbacks,
            verbose=1)
        # fit threshold
        thres_titles, thres_contents, thres_Y0, thres_Y1 = [], [], [], []
        for _ in range(10):
            x_dict, y_dict = data_generator.next()
            thres_titles.extend(x_dict['title'].tolist())
            thres_contents.extend(x_dict['content'].tolist())
            thres_Y0.extend(y_dict['Y0'].tolist())
            thres_Y1.extend(y_dict['Y1'].tolist())
        thres_dataset = {
            'title': np.array(thres_titles),
            'content': np.array(thres_contents),
            'Y0': np.array(thres_Y0),
            'Y1': np.array(thres_Y1)
        }
        self.model.fit_thresholds(
            thres_dataset,
            validation_data=valid_data,
            top_k=None,
            alpha=np.logspace(-3, 3, num=10).tolist(),
            verbose=1,
            batch_size=self.params['iter']['batch_size'],
            use_hidden_feature=True,
            vocab_size=len(vocabulary))
        # 回收内存
        del valid_data
        del thres_dataset
        gc.collect()
        # test model
        if test_data_file is None:
            return True
        test_data_generator = generate_arrays_from_dataset(
            test_data_file,
            vocabulary,
            category,
            self.params['Y0']['dim'],
            split_tag='@@@',
            lbl_text_index=[0, 1, 2],
            batch_size=self.params['iter']['batch_size'],
            title_sequence_length=self.params['title_layer'][
                'sequence_length'],
            content_sequence_length=self.params['content_layer'][
                'sequence_length'],
            use_G1=True,
            padding_word='<PAD/>')
        for i in range(self.params['iter']['test_steps']):
            print('predict %s / %s....start loading batch_data' %
                  (i, self.params['iter']['test_steps']))
            x_dict, y_dict = test_data_generator.next()
            batch_data = {
                'title': x_dict['title'],
                'content': x_dict['content'],
                'Y0': y_dict['Y0'],
                'Y1': y_dict['Y1']
            }
            print('start predicting.....')
            probs, preds = self.model.predict_combine(
                batch_data,
                verbose=0,
                batch_size=self.params['iter']['batch_size'],
                use_hidden_feature=True)
            print('current %i batch_data predicted completed.' % i)
            target = np.hstack([batch_data[k] for k in ['Y0', 'Y1']])
            pred = np.hstack([preds[k] for k in ['Y0', 'Y1']])
            targets_all = target if i == 0 else np.vstack((targets_all,
                                                           target))
            preds_all = pred if i == 0 else np.vstack((preds_all, pred))
        # save predict sampless
        # save_predict_samples(
        #     raw_test_dataset, test_dataset, preds_all, save_num=2000)
        for i in range(100):
            print('\n')
            print(' '.join([
                vocabulary_inv[ii] for ii in batch_data['title'][i]
            ]) + '@@@' + ' '.join(
                [vocabulary_inv[ii] for ii in batch_data['content'][i]]))
            print(np.where(target[i] == 1))
            print(
                ' '.join([category[ii] for ii in np.where(target[i] == 1)[0]]))
            print(np.where(pred[i] == 1))
            print(' '.join([category[ii] for ii in np.where(pred[i] == 1)[0]]))

        # print('start calculate confuse matix....')
        # get_confuse(batch_data, preds, 'Y0')
        # get_confuse(batch_data, preds, 'Y1')

        hl = hamming_loss(batch_data, preds)
        f1_macro = f1_measure(batch_data, preds, average='macro')
        f1_micro = f1_measure(batch_data, preds, average='micro')
        f1_samples = f1_measure(batch_data, preds, average='samples')
        p_at_1 = precision_at_k(batch_data, probs, K=1)
        p_at_3 = precision_at_k(batch_data, probs, K=3)
        p_at_5 = precision_at_k(batch_data, probs, K=5)

        for k in ['Y0', 'Y1', 'all']:
            print
            print("Hamming loss (%s): %.2f" % (k, hl[k]))
            print("F1 macro (%s): %.4f" % (k, f1_macro[k]))
            print("F1 micro (%s): %.4f" % (k, f1_micro[k]))
            print("F1 sample (%s): %.4f" % (k, f1_samples[k]))
            print("P@1 (%s): %.4f" % (k, p_at_1[k]))
            print("P@3 (%s): %.4f" % (k, p_at_3[k]))
            print("P@5 (%s): %.4f" % (k, p_at_5[k]))

        t_recall, t_precision, t_f1 = recall_precision_f1(
            targets_all, preds_all)
        # t_recall, t_precision = all_recall_precision(test_dataset['Y1'], preds['Y1'])
        print('total recall : %.4f' % t_recall)
        print('total precision : %.4f' % t_precision)
        print('total f1 : %.4f\n' % t_f1)

        g1_recall, g1_precision, g1_f1 = recall_precision_f1(
            targets_all[:, :self.params['Y0']['dim']],
            preds_all[:, :self.params['Y0']['dim']])
        print('G1 recall : %.4f' % g1_recall)
        print('G1 precision : %.4f' % g1_precision)
        print('G1 f1 : %.4f\n' % g1_f1)

        g2_recall, g2_precision, g2_f1 = recall_precision_f1(
            targets_all[:, self.params['Y0']['dim']:],
            preds_all[:, self.params['Y0']['dim']:])
        print('G2 recall : %.4f' % g2_recall)
        print('G2 precision : %.4f' % g2_precision)
        print('G2 f1 : %.4f\n' % g2_f1)

    def predict(self):
        pass


# def get_confuse(data, pred, kw):
#     y_true, y_pre = [], []
#     _cate = Y0 if kw == 'Y0' else Y1
#     for i in range(len(data[kw])):
#         y_true.append([_cate[ii] for ii in np.where(data[kw][i] == 1)[0]])
#         y_pre.append([_cate[ii] for ii in np.where(pred[kw][i] == 1)[0]])
#     confuse_dict = ml_confuse(y_true, y_pre)
#     with open('../docs/CNN/%s_confuse' % kw, 'w') as f:
#         for lbl, c_dict in confuse_dict.items():
#             c_sort = sorted(c_dict.items(), key=lambda d: d[1], reverse=True)
#             cont = ' '.join(map(lambda x: '%s:%s' % x, c_sort)) + '\n'
#             f.write('%s\t%s' % (lbl, cont))


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
