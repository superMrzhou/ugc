"""
Utility functions for constructing MLC models.
"""
import keras.backend as K
from keras.layers import Conv1D, MaxPool1D, Embedding, Flatten
from keras.layers import Dense, Dropout, Input, Lambda, Reshape
from keras.layers import ActivityRegularization, concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from utils.models import MLC


def assemble(name, params):
    if name == 'MLP':
        return assemble_mlp(params)
    elif name == 'ADIOS':
        return assemble_adios(params)
    else:
        raise ValueError("Unknown name of the model: %s." % name)


def assemble_mlp(params):
    """
    Construct an MLP model of the form:
                                X-H-H1-Y
    where all the H-layers are optional and depend on whether they are
    specified in the params dictionary.
    """

    # X
    X = Input(shape=(params['X']['dim'],), dtype='float32', name='X')

    # embedding
    if 'embedding_size' in params['X'] and params['X']['embedding_size'] != None:
        X = Embedding(output_dim=params['X']['embedding_size'],
                      input_dim=params['X']['vocab_size'],
                      input_length=params['X']['dim'])(X)

    # Conv and max-pooling
    filters = params['Conv2D']['filters']
    pooled_output = []
    for size in params['Conv2D']['filter_size']:
        conv = Conv1D(filters,
                      (size, size),
                      padding='valid',
                      activation='relu'
                      )(X)
        pooling = MaxPool1D((params['X']['dim'] - size, 1))(conv)
        flatten = Flatten()(pooling)
        pooled_output.append(flatten)

    # combine all the pooled feature
    H0 = concatenate(pooled_output)

    # batch_norm
    if 'batch_norm' in params['H'] and params['H']['batch_norm'] != None:
        H0 = BatchNormalization(name='H0_batchNorm', **
                                params['H']['batch_norm'])(H0)

    # dropout
    if 'dropout' in params['H']:
        H0 = Dropout(params['H']['dropout'], name='H0_dropout')(H0)

    # H1
    if 'H1' in params:
        kwargs = params['H1']['kwargs'] if 'kwargs' in params['H1'] else {}
        # Relu
        H1 = Dense(params['H1']['dim'],
                   activation='relu',
                   **kwargs)(H0)

        # batch_norm
        if 'batch_norm' in params['H1'] and params['H1']['batch_norm'] != None:
            H1 = BatchNormalization(
                name='H1_batchNorm', **params['H1']['batch_norm'])(H1)

        # dropout
        if 'dropout' in params['H1']:
            H1 = Dropout(params['H1']['dropout'], name='H1_dropout')(H1)

    # Y
    kwargs = params['Y']['kwargs'] if 'kwargs' in params['Y'] else {}
    if 'W_regularizer' in kwargs:
        kwargs['W_regularizer'] = l2(kwargs['W_regularizer'])
    # sigmoid
    Y = Dense(params['Y']['dim'],
              activation='sigmoid',
              **kwargs)(H1)

    if 'activity_reg' in params['Y']:
        Y = ActivityRegularization(name='main_output',
                                   **params['Y']['activity_reg']
                                   )(Y)

    return MLC(inputs=X, outputs=Y)


def assemble_adios(params):
    """
    Construct one of the ADIOS models. The general structure is the following:
                                X-H-(Y0|H0)-H1-Y1,
    where all the H-layers are optional and depend on whether they are
    specified in the params dictionary.
    """
    # X

    input_shape = (params['X']['sequence_length'], params['X']['embedding_dim']
                   ) if params['iter']['model_type'] == "CNN-static" else (params['X']['sequence_length'],)
    X = Input(shape=input_shape, dtype='float32', name='X')

    # embedding
    # Static model do not have embedding layer
    if params['iter']['model_type'] == "CNN-static":
        embedding = X
    elif 'embedding_dim' in params['X'] and params['X']['embedding_dim'] != None:
        embedding = Embedding(output_dim=params['X']['embedding_dim'],
                              input_dim=params['X']['vocab_size'],
                              input_length=params['X']['sequence_length'],
                              name="embedding",
                              mask_zero=False
                              )(X)
    else:
        exit('embedding_dim param is not given!')

    # expanding dimension
    #embed_reshape = Reshape((params['X']['sequence_length'], params['X']['embedding_dim'], 1))(embedding)

    # Conv and max-pooling
    filters = params['Conv1D']['filters']
    pooled_output = []
    for size in params['Conv1D']['filter_size']:
        conv = Conv1D(filters=filters,
                      kernel_size=size,
                      padding='valid',
                      activation='relu',
                      strides=1
                      )(embedding)
        pooling = MaxPool1D(pool_size=params['Conv1D']['pooling_size'])(conv)
        flatten = Flatten()(pooling)
        pooled_output.append(flatten)

    # combine all the pooled feature as the hidden layer between X and Y0
    H = concatenate(pooled_output) if len(
        pooled_output) > 1 else pooled_output[0]

    # batch_norm
    if 'batch_norm' in params['H'] and params['H']['batch_norm'] != None:
        H = BatchNormalization(name='H_batchNorm',
                               **params['H']['batch_norm'])(H)

    # dropout
    if 'dropout' in params['H']:
        H = Dropout(params['H']['dropout'], name="H_dropout")(H)

    # Y0 output
    kwargs = params['Y0']['kwargs'] if 'kwargs' in params['Y0'] else {}
    if 'W_regularizer' in kwargs:
        kwargs['W_regularizer'] = l2(kwargs['W_regularizer'])
    Y0 = Dense(params['Y0']['dim'],
               activation='sigmoid',
               name='Y0_active',
               **kwargs)(H)
    if 'activity_reg' in params['Y0']:
        Y0 = ActivityRegularization(name='Y0_output',
                                    **params['Y0']['activity_reg']
                                    )(Y0)
    # batch_norm
    if 'batch_norm' in params['Y0'] and params['Y0']['batch_norm'] != None:
        Y0 = BatchNormalization(name='Y0',
                                **params['Y0']['batch_norm']
                                )(Y0)

    # H0
    if 'H0' in params:  # we have a composite layer (Y0|H0)
        kwargs = params['H0']['kwargs'] if 'kwargs' in params['H0'] else {}

        # ReLu
        H0 = Dense(params['H0']['dim'],
                   activation='relu',
                   **kwargs)(H)
        # batch_norm
        if 'batch_norm' in params['H0'] and params['H0']['batch_norm'] != None:
            H0 = BatchNormalization(name='H0_batchNorm',
                                    **params['H0']['batch_norm']
                                    )(H0)
        # dropout
        if 'dropout' in params['H0']:
            H0 = Dropout(params['H0']['dropout'],
                         name='H0_dropout')(H0)

        Y0_H0 = concatenate([Y0, H0])
    else:
        Y0_H0 = Y0

    # H1
    if 'H1' in params:  # there is a hidden layer between Y0 and Y1
        kwargs = params['H1']['kwargs'] if 'kwargs' in params['H1'] else {}

        H1 = Dense(params['H1']['dim'],
                   activation='relu',
                   **kwargs)(Y0_H0)
        # batch_norm
        if 'batch_norm' in params['H1'] and params['H1']['batch_norm'] != None:
            H1 = BatchNormalization(name='H1_batch_norm',
                                    **params['H1']['batch_norm']
                                    )(H1)
        # dropout
        if 'dropout' in params['H1']:
            H1 = Dropout(params['H1']['dropout'],
                         name='H1_dropout')(H1)
    else:
        H1 = Y0_H0

    # Y1
    kwargs = params['Y1']['kwargs'] if 'kwargs' in params['Y1'] else {}
    if 'W_regularizer' in kwargs:
        kwargs['W_regularizer'] = l2(kwargs['W_regularizer'])
    Y1 = Dense(params['Y1']['dim'],
               activation='sigmoid',
               name='Y1_activation',
               **kwargs)(H1)

    if 'activity_reg' in params['Y0']:
        Y1 = ActivityRegularization(name='Y1',
                                    **params['Y1']['activity_reg']
                                    )(Y1)

    return MLC(inputs=X, outputs=[Y0, Y1])
