"""
Utility functions for constructing MLC models.
"""
from keras.layers import Conv1D, Embedding, Flatten, AvgPool1D, MaxPool1D
from keras.layers import Dense, Dropout, Input, Activation
from keras.layers import ActivityRegularization, concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers import GRU

from utils.models import MLC
# from utils.selu import selu, dropout_selu


def assemble(name, params):
    if name == 'ADIOS':
        return assemble_adios(params)
    else:
        raise ValueError("Unknown name of the model: %s." % name)


def assemble_adios(params):
    """
    Construct one of the ADIOS models. The general structure is the following:
                                X-H-(Y0|H0)-H1-Y1,
    where all the H-layers are optional and depend on whether they are
    specified in the params dictionary.
    """
    # X

    input_shape = (params['X']['sequence_length'], params['X']['embedding_dim']
                   ) if params['iter']['model_type'] == "CNN-static" else (
                       params['X']['sequence_length'], )
    X = Input(shape=input_shape, dtype='int32', name='X')

    # embedding
    # Static model do not have embedding layer
    if params['iter']['model_type'] == "CNN-static":
        embedding = BatchNormalization()(X)
    elif 'embedding_dim' in params['X'] and params['X']['embedding_dim']:
        embedding = Embedding(
            output_dim=params['X']['embedding_dim'],
            input_dim=params['X']['vocab_size'],
            input_length=params['X']['sequence_length'],
            name="embedding",
            mask_zero=False)(X)
        # embedding = BatchNormalization()(embedding)
    else:
        exit('embedding_dim param is not given!')

    # expanding dimension
    # embed_reshape = Reshape((params['X']['sequence_length'], params['X']['embedding_dim'], 1))(embedding)

    # multi-layer Conv and max-pooling
    conv_layer_num = len(params['Conv1D'])
    for i in range(1, conv_layer_num+1):
        H_input = embedding if i == 1 else H
        conv = Conv1D(
            filters=params['Conv1D']['layer%s' % i]['filters'],
            kernel_size=params['Conv1D']['layer%s' % i]['filter_size'],
            padding=params['Conv1D']['layer%s' % i]['padding_mode'],
            # activation='relu',
            strides=1,
            bias_regularizer=l2(0.01))(H_input)
        conv_batch_norm = Activation('relu')(BatchNormalization()(conv))
        # conv_batch_norm = selu(BatchNormalization()(conv))
        pool_size = params['Conv1D']['layer%s' % i]['pooling_size']
        conv_pooling = MaxPool1D(pool_size=pool_size)(conv_batch_norm)
        # dropout
        if 'dropout' in params['Conv1D']['layer%s' % i]:
            H = Dropout(params['Conv1D']['layer%s' % i]['dropout'])(conv_pooling)
            # H = dropout_selu(conv_pooling, params['Conv1D']['layer%s' % i]['dropout'])

    # flatten
    H = Flatten(name='H')(H)
    # H = Dropout(0.3, name='H')(BatchNormalization()(GRU(128)(H)))

    # Y0 output
    kwargs = params['Y0']['kwargs'] if 'kwargs' in params['Y0'] else {}
    if 'W_regularizer' in kwargs:
        kwargs['W_regularizer'] = l2(kwargs['W_regularizer'])
    Y0 = Dense(
        params['Y0']['dim'],
        # activation='sigmoid',
        name='Y0_active',
        bias_regularizer=l2(0.01),
        **kwargs)(H)
    # batch_norm
    if 'batch_norm' in params['Y0']:
        Y0 = BatchNormalization(**params['Y0']['batch_norm'])(Y0)
    Y0 = Activation('softmax')(Y0)
    if 'activity_reg' in params['Y0']:
        Y0 = ActivityRegularization(
            name='Y0', **params['Y0']['activity_reg'])(Y0)

    # H0
    if 'H0' in params:  # we have a composite layer (Y0|H0)
        kwargs = params['H0']['kwargs'] if 'kwargs' in params['H0'] else {}

        # ReLu
        H0 = Dense(
            params['H0']['dim'],
            # activation='relu',
            bias_regularizer=l2(0.01),
            **kwargs)(H)
        # batch_norm
        if 'batch_norm' in params['H0'] and params['H0']['batch_norm']:
            H0 = BatchNormalization(
                name='H0_batchNorm', **params['H0']['batch_norm'])(H0)
        H0 = Activation('relu')(H0)
        # dropout
        if 'dropout' in params['H0']:
            H0 = Dropout(params['H0']['dropout'], name='H0_dropout')(H0)
        Y0_H0 = concatenate([Y0, H0])
        # Y0_H0 = H0
    else:
        Y0_H0 = Y0

    # H1
    if 'H1' in params:  # there is a hidden layer between Y0 and Y1
        kwargs = params['H1']['kwargs'] if 'kwargs' in params['H1'] else {}

        H1 = Dense(
            params['H1']['dim'],
            # activation='relu',
            bias_regularizer=l2(0.01),
            **kwargs)(Y0_H0)
        # batch_norm
        if 'batch_norm' in params['H1'] and params['H1']['batch_norm']:
            H1 = BatchNormalization(
                name='H1_batch_norm', **params['H1']['batch_norm'])(H1)
        H1 = Activation('relu')(H1)
        # dropout
        if 'dropout' in params['H1']:
            H1 = Dropout(params['H1']['dropout'], name='H1_dropout')(H1)
    else:
        H1 = Y0_H0

    # Y1
    kwargs = params['Y1']['kwargs'] if 'kwargs' in params['Y1'] else {}
    if 'W_regularizer' in kwargs:
        kwargs['W_regularizer'] = l2(kwargs['W_regularizer'])
    Y1 = Dense(
        params['Y1']['dim'],
        # activation='sigmoid',
        name='Y1_activation',
        bias_regularizer=l2(0.01),
        **kwargs)(H1)
    # batch_norm
    if 'batch_norm' in params['Y1']:
        Y1 = BatchNormalization(**params['Y1']['batch_norm'])(Y1)
    Y1 = Activation('softmax')(Y1)

    if 'activity_reg' in params['Y1']:
        Y1 = ActivityRegularization(
            name='Y1', **params['Y1']['activity_reg'])(Y1)

    return MLC(inputs=X, outputs=[Y0, Y1])
