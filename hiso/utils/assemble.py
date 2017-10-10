"""
Utility functions for constructing MLC models.
"""
import tensorflow as tf
from keras.layers import Embedding, Dense, Dropout, concatenate, GRU
from keras.layers.wrappers import Bidirectional
from keras.layers import ActivityRegularization
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.objectives import categorical_crossentropy

from utils.models import MLC


def assemble(name, params):
    if name == 'ADIOS':
        return assemble_adios(params)
    else:
        raise ValueError("Unknown name of the model: %s." % name)


class HISO(object):
    def __init__(self, sentence_len, embed_size, vocab_size, Y0_dim, Y1_dim, mask_zero=True):
        # input
        inputs = tf.placeholder(tf.float32, [None, sentence_len], name='input')
        # 1.base layers: embedding
        embedding = Embedding(output_dim=embed_size, input_dim=vocab_size, input_length=sentence_len, mask_zero=mask_zero, name='embedding')(inputs)
        # 2. semantic layers: Bidirectional GRU
        Bi_GRU = Bidirectional(GRU(64), merge_mode='concat', name='Bi_GRU')(embedding)

        # 3. middle layer for predict Y0
        Y0_preds = Dense(Y0_dim, activation='softmax', name='Y0_predictions')(Bi_GRU)
        Y0 = tf.placeholder(tf.float32, [None, Y0_dim], name='Y0')
        # 4. upper hidden layers
        # Firstly, learn a hidden layer from Bi_GRU
        # Secondly, consider Y0_preds as middle feature and combine it with hidden layer
        hidden_layer_1 = Dense(64, name='hidden_layer_1')(Bi_GRU)
        combine_layer = concatenate([Y0_preds, hidden_layer_1], axis=-1, name='combine_layer')
        hidden_layer_2 = Dense(96, name='hidden_layer_2')(combine_layer)
        # 5. layer for predict Y1
        Y1_preds = Dense(Y1_dim, activation='softmax', name='Y1_predictions')(hidden_layer_2)
        Y1 = tf.placeholer(tf.float32, [None, Y1_dim], name='Y1')

        # Calculate loss
        with tf.name_scope('loss'):
            Y0_loss = tf.reduce_mean(categorical_crossentropy(Y0, Y0_preds), name='Y0_loss')
            Y1_loss = tf.reduce_mean(categorical_crossentropy(Y1, Y1_preds), name='Y1_loss')
            loss = tf.add(Y0_loss, Y1_loss, name='loss')

    def hamming_loss(self, data, preds):
        # Compute the scores for each output separately
        hl = {k: float((data[k] != preds[k]).sum(axis=1).mean()) for k in preds}

        # Concatenate the outputs and compute the overall score
        targets_all = np.hstack([data[k] for k in preds])
        preds_all = np.hstack([preds[k] for k in preds])
        hl['all'] = float((targets_all != preds_all).sum(axis=1).mean())

    def rank_loss(self, labels, preds):
        pass

    def one_error(self, labels, preds):
        pass

    def Coverage(self, labels, preds):
        pass

    def average_precision(self, labels, preds):
        pass
