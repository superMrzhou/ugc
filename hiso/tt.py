import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, multiply
from keras.objectives import categorical_crossentropy
from utils.metrics import Hamming_loss, Coverage, F1_measure, One_error, Ranking_loss

sess = tf.Session()
K.set_session(sess)

np.random.seed(1337)  # for reproducibility


class test(object):
    def __init__(self, input_dim, y_dim):
        self.inputs = tf.placeholder(tf.float32, shape=[None, input_dim])

        attention_probs = Dense(
                    input_dim, activation='softmax', name='attention_vec')(self.inputs)
        attention_mul = multiply([self.inputs, attention_probs])
        attention_mul = Dense(64)(attention_mul)
        self.probs = Dense(y_dim, activation='sigmoid')(attention_mul)
        self.labels = tf.placeholder(tf.float32, shape=[None, y_dim])

        self.loss = tf.reduce_mean(categorical_crossentropy(self.labels, self.probs))
        # self.acc_value = tf.reduce_mean(categorical_accuracy(self.labels, preds))
        self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)


def get_data(n, input_dim, y_dim, attention_column=1):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    x = np.random.standard_normal(size=(n, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, y_dim))
    for i in range(y_dim):
        x[:, i*3] = y[:, i]
    return x, y


def do_eval(labels, probs):
    preds = probs > 0.5
    return {'hamming_loss': Hamming_loss(labels, preds),
            'ranking_loss': Ranking_loss(labels, probs),
            'f1@micro': F1_measure(labels, preds, average='micro'),
            'one_error': One_error(labels, probs),
            'coverage': Coverage(labels, probs)}


if __name__ == '__main__':
    N = 1000
    input_dim = 32
    y_dim = 6
    inputs, outputs = get_data(N, input_dim, y_dim)

    # run model
    with sess.as_default():
        tt = test(input_dim=input_dim, y_dim=y_dim)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # feed data to training
        number_of_training_data = len(outputs)
        batch_size = 20
        for epoch in range(3):
            for start, end in zip(
                    range(0, number_of_training_data, batch_size),
                    range(batch_size, number_of_training_data, batch_size)):
                trn_loss, trn_probs, _ = sess.run(
                        [tt.loss, tt.probs, tt.train_step],
                        feed_dict={
                            tt.inputs: inputs[start:end],
                            tt.labels: outputs[start:end],
                        })
                print('epoch: {}, train loss: {}'.format(epoch, trn_loss))
                print(do_eval(np.array(outputs[start:end]), trn_probs))
