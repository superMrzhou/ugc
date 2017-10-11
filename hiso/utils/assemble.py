"""
Utility functions for constructing MLC models.
"""
import numpy as np
import tensorflow as tf
from keras.layers import GRU, Dense, Embedding, concatenate
from keras.layers.wrappers import Bidirectional
from keras.metrics import categorical_accuracy
from keras.objectives import categorical_crossentropy


class HISO(object):
    def __init__(self,
                 sentence_len,
                 embed_size,
                 vocab_size,
                 Y0_dim,
                 Y1_dim,
                 mask_zero=True):
        # input
        self.inputs = tf.placeholder(
            tf.float32, [None, sentence_len], name='input')
        # 1.base layers: embedding
        embedding = Embedding(
            output_dim=embed_size,
            input_dim=vocab_size,
            input_length=sentence_len,
            mask_zero=mask_zero,
            name='embedding')(self.inputs)
        # 2. semantic layers: Bidirectional GRU
        Bi_GRU = Bidirectional(
            GRU(64), merge_mode='concat', name='Bi_GRU')(embedding)

        # 3. middle layer for predict Y0
        Y0_preds = Dense(
            Y0_dim, activation='softmax', name='Y0_predictions')(Bi_GRU)
        self.Y0 = tf.placeholder(tf.float32, [None, Y0_dim], name='Y0')
        # 4. upper hidden layers
        # Firstly, learn a hidden layer from Bi_GRU
        # Secondly, consider Y0_preds as middle feature and combine it with hidden layer
        hidden_layer_1 = Dense(64, name='hidden_layer_1')(Bi_GRU)
        combine_layer = concatenate(
            [Y0_preds, hidden_layer_1], axis=-1, name='combine_layer')
        hidden_layer_2 = Dense(96, name='hidden_layer_2')(combine_layer)
        # 5. layer for predict Y1
        Y1_preds = Dense(
            Y1_dim, activation='softmax',
            name='Y1_predictions')(hidden_layer_2)
        self.Y1 = tf.placeholder(tf.float32, [None, Y1_dim], name='Y1')

        # Calculate loss
        with tf.name_scope('loss'):
            Y0_loss = tf.reduce_mean(
                categorical_crossentropy(self.Y0, Y0_preds), name='Y0_loss')
            Y1_loss = tf.reduce_mean(
                categorical_crossentropy(self.Y1, Y1_preds), name='Y1_loss')
            self.loss = tf.add_n([Y0_loss, Y1_loss], name='loss')

        self.train_op = tf.train.GradientDescentOptimizer(0.5).minimize(
            self.loss)

    def hamming_loss(self, labels, preds):
        '''
        用于度量样本在单个标记上的真实标记和预测标记的错误匹配情况
        @labels: true labels of samples
        @preds:  predict labels of samples
        '''
        hl = tf.reduce_mean(
            tf.not_equal(tf.cast(labels, tf.int32), tf.cast(preds, tf.int32)))
        return hl

    def rank_loss(self, labels, preds):
        '''
        用来考察样本的不相关标记的排序低于相关标记的排序情况
        '''
        pass

    def one_error(self, labels, probs):
        '''
        用来考察预测值排在第一位的标记却不隶属于该样本的情况
        @labels: true labels of samples
        @probs:  label's probility  of samples
        '''
        # get the index with the largest value across axes of a Tensor
        preds_at_1 = tf.argmax(probs, axis=1)
        labels_at_1 = tf.gather(labels, preds_at_1, axis=1)
        error = tf.reduce_mean(
            tf.not_equal(
                tf.cast(labels_at_1, tf.int32), tf.cast(preds_at_1, tf.int32)))
        return error

    def Coverage(self, labels, probs):
        '''
        用于度量平均上需要多少步才能遍历样本所有的相关标记
        @labels: true labels of samples
        @probs:  label's probility  of samples
        '''
        # find min prob of true label
        lbl_probs_min = tf.reduce_min(tf.multiply(labels, probs), axis=1)
        # [n, 1] X [1, L] = [n, L]
        min_probs = tf.matmul(lbl_probs_min, tf.constant(1, shape=[1, tf.shape(probs)[-1]]))

        steps = tf.reduce_mean(tf.greater_equal(probs, min_probs))
        return steps

    def average_precision(self, labels, preds):
        '''
        用来考察排在隶属于该样本标记之前标记仍属于样本的相关标记集合的情况
        '''
        pass
