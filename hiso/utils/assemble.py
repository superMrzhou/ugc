"""
Utility functions for constructing MLC models.
multilabel loss functions see：
http://d0evi1.com/sklearn/model_evaluation/ and http://www.jos.org.cn/html/2014/9/4634.htm
"""
import tensorflow as tf
from keras.layers import GRU, Dense, Embedding, concatenate
from keras.layers import Dropout, BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.objectives import binary_crossentropy


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
            GRU(64, dropout=0.3, recurrent_dropout=0.3), merge_mode='concat', name='Bi_GRU')(embedding)
        Bi_GRU = BatchNormalization(momentum=0.9, name='Bi_GRU_BN')(Bi_GRU)
        # 3. middle layer for predict Y0
        self.Y0_preds = Dense(
            Y0_dim, activation='sigmoid', name='Y0_predictions')(Bi_GRU)
        self.Y0 = tf.placeholder(tf.float32, [None, Y0_dim], name='Y0')
        # 4. upper hidden layers
        # Firstly, learn a hidden layer from Bi_GRU
        # Secondly, consider Y0_preds as middle feature and combine it with hidden layer
        hidden_layer_1 = Dense(64, name='hidden_layer_1')(Bi_GRU)
        hidden_layer_1 = BatchNormalization(momentum=0.9, name='hidden_layer_1_BN')(hidden_layer_1)
        hidden_layer_1 = Dropout(0.3, name='hidden_layer_1_dropout')(hidden_layer_1)

        combine_layer = concatenate(
            [self.Y0_preds, hidden_layer_1], axis=-1, name='combine_layer')

        hidden_layer_2 = Dense(96, name='hidden_layer_2')(combine_layer)
        hidden_layer_2 = BatchNormalization(momentum=0.9, name='hidden_layer_2_BN')(hidden_layer_2)
        hidden_layer_2 = Dropout(0.3, name='hidden_layer_2_dropout')(hidden_layer_2)

        # 5. layer for predict Y1
        self.Y1_preds = Dense(
            Y1_dim, activation='sigmoid',
            name='Y1_predictions')(hidden_layer_2)
        self.Y1 = tf.placeholder(tf.float32, [None, Y1_dim], name='Y1')

        # Calculate loss
        with tf.name_scope('loss'):
            Y0_loss = tf.reduce_mean(
                binary_crossentropy(self.Y0, self.Y0_preds), name='Y0_loss')
            Y1_loss = tf.reduce_mean(
                binary_crossentropy(self.Y1, self.Y1_preds), name='Y1_loss')
            self.loss = tf.add_n([Y0_loss, Y1_loss], name='loss')

        self.train_op = tf.train.RMSPropOptimizer(1e-2).minimize(
            self.loss)
