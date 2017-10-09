"""Hierarachy lstm model for multi-labels classifictaion."""

from keras.layers import Masking
from keras.layers import Input, LSTM, Dense, concatenate, Embedding, Activation
from keras.layers import MaxPool1D, AvgPool1D, Conv1D, Flatten, Dropout
from keras.models import Model

from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from keras.metrics import categorical_accuracy


def assemble_multi_lstm(params):
    """Construct Hierarachy lstm model by params.

    Description usually contains more detail and abstract information about sub_labels
    in deep architecture. We combine inner output with description hidden output
    to predict sub_labels
    """
    title_wd_input_shape = (
        params['title']['word']['sequence_length'],
        # params['title']['word']['embedding_dim']
    )
    # input
    title_wd_input = Input(shape=title_wd_input_shape, dtype='int32', name='title_wd_input')
    # embeding and filter padding value of index=0
    embedder = Embedding(
        output_dim=params['title']['word']['embedding_dim'],
        input_dim=params['title']['word']['vocab_size'],
        input_length=params['title']['word']['sequence_length'],
        name="embedding",
        mask_zero=True)
    # filter mask_value
    # title_wd_input = Masking(
    #     mask_value=params['iter']['mask_value'],
    #     # input_shape=title_wd_input_shape,
    #     # dtype='float32',
    #     name='title_wd_input')(X)
    title_wd_embedding = embedder(title_wd_input)
    # batch norm
    title_wd_input_batch_norm = BatchNormalization(momentum=0.9)(title_wd_embedding)
    # first layer
    title_wd_first_lstm_out = LSTM(params['lstm']['title_cell'], return_sequences=True)(title_wd_input_batch_norm)
    # batch norm
    title_wd_first_lstm_batch_norm = BatchNormalization(momentum=0.9)(title_wd_first_lstm_out)
    title_wd_first_lstm_out = Dropout(0.3)(title_wd_first_lstm_batch_norm)

    # second layer
    title_wd_second_lstm_out = LSTM(params['lstm']['title_cell'])(title_wd_first_lstm_out)
    # batch norm
    title_wd_second_lstm_batch_norm = BatchNormalization(momentum=0.9)(title_wd_second_lstm_out)
    title_wd_second_lstm_out = Dropout(0.3)(title_wd_second_lstm_batch_norm)
    # title char
    # title_char_input_shape = (
    #     params['title']['char']['sequence_length'],
    #     params['title']['char']['embedding_dim']
    # )
    # # filter mask_value
    # title_char_input = Masking(
    #     mask_value=params['iter']['mask_value'],
    #     input_shape=title_char_input_shape,
    #     dtype='float32',
    #     name='title_char_input')
    # title_char_lstm_out = LSTM(params['lstm']['title_cell'])(title_char_input)

    # desc word
    desc_wd_input_shape = (
        params['desc']['word']['sequence_length'],
        # params['desc']['word']['embedding_dim']
    )
    # input
    desc_wd_input = Input(shape=desc_wd_input_shape, dtype='int32', name='desc_wd_input')
    # embeding and filter padding value of index=0
    desc_wd_embedding = embedder(desc_wd_input)
    # filter mask_value
    # desc_wd_input = Masking(
    #     mask_value=params['iter']['mask_value'],
    #     input_shape=title_wd_input_shape,
    #     dtype='float32',
    #     name='title_wd_input')
    # batch norm
    # desc_wd_input_batch_norm = BatchNormalization(momentum=0.9)(desc_wd_embedding)
    desc_wd_first_lstm_out = LSTM(params['lstm']['desc_cell'], return_sequences=True)(desc_wd_embedding)
    # batch norm
    desc_wd_first_lstm_batch_norm = BatchNormalization(momentum=0.9)(desc_wd_first_lstm_out)
    desc_wd_first_lstm_out = Dropout(0.3)(desc_wd_first_lstm_batch_norm)

    # second layer
    desc_wd_second_lstm_out = LSTM(params['lstm']['desc_cell'])(desc_wd_first_lstm_out)
    # batch norm
    desc_wd_second_lstm_batch_norm = BatchNormalization(momentum=0.9)(desc_wd_second_lstm_out)
    desc_wd_second_lstm_out = Dropout(0.3)(desc_wd_second_lstm_batch_norm)
    # # desc char
    # desc_char_input_shape = (
    #     params['desc']['char']['sequence_length'],
    #     params['desc']['char']['embedding_dim']
    # ) if params['iter']['embed_type'] == "static" else (
    #     params['desc']['char']['sequence_length'], )
    # # filter mask_value
    # desc_char_input = Masking(
    #     mask_value=params['mask_value'],
    #     input_shape=desc_char_input_shape,
    #     dtype='float32',
    #     name='desc_char_input')
    # desc_char_lstm_out = LSTM(params['lstm']['desc_cell'])(desc_char_input)

    # Y0 predicted by title
    # title_vec = concatenate(
    #     [title_wd_lstm_out, title_char_lstm_out], name='title_vec')
    # # cells of combined lstm default by individual times two
    # title_lstm_out = LSTM(
    #     int(params['lstm']['ratio'] * params['lstm']['title_cell']))(title_vec)

    Y0 = Dense(params['Y0']['dim'])(title_wd_second_lstm_out)
    # batch norm
    Y0_batch_norm = BatchNormalization(momentum=0.9)(Y0)
    Y0_output = Activation(params['Y0']['activation'], name='Y0')(Y0_batch_norm)

    # combine desc-char and desc-wd and put in LSTM cell
    # desc_vec = concatenate(
    #     [desc_char_lstm_out, desc_word_lstm_out], name='desc_vec')
    # # cells of combined lstm default by individual times two
    # desc_lstm_out = LSTM(
    #     int(params['lstm']['ratio'] * params['lstm']['desc_cell']))(desc_vec)
    #
    # # assume label level information as abstract feature
    # total_vec = concatenate([Y0_output, desc_lstm_out])
    # total_lstm_out = LSTM(params['lstm']['deep_lstm_cell'])(
    #     total_vec) if params['lstm']['is_deep_lstm'] == 1 else total_vec
    Y1_input = concatenate([title_wd_second_lstm_out, desc_wd_second_lstm_out])
    Y1_output = Dense(params['Y1']['dim'])(Y1_input)
    # batch norm
    Y1_batch_norm = BatchNormalization(momentum=0.9)(Y1_output)
    Y1_output = Activation(params['Y1']['activation'])(Y1_batch_norm)

    model = Model(
        inputs=[
            title_wd_input, desc_wd_input
        ],
        outputs=[Y0_output, Y1_output])
    model.compile(
        optimizer=params['iter']['optimizer'],
        loss={
            'Y0_output': params['Y0']['loss'],
            'Y1_output': params['Y1']['loss']
        },
        loss_weights={
            'Y0_output': params['Y0']['loss_weight'],
            'Y1_output': params['Y1']['loss_weight']
        })
    return model


def assemble_cnn_lstm(params):
    """Construct Hierarachy lstm model by params and inner feature selected by CNN.

    Every word or char was looked up in word2vec metrix. Then conv layer was applied to selected
    features from neighbor words. LSTM also used to keep memory information availabel in the
    learning process.
    """
    # title char
    title_char_input = Input(
        shape=(params['title']['char']['sequence_length'],
               params['title']['char']['embedding_dim']),
        dtype='float32',
        name='title_char_input')
    # conv1D layer
    title_char_cnn_out = keras_conv(
        title_char_input,
        filters=params['CNN']['title_filters'],
        size_list=params['CNN']['title_sizes'],
        pooling_size=params['CNN']['pooling_size'],
        isMaxpool=params['CNN']['isMaxpool'])
    # LSTM cell
    title_char_lstm_out = LSTM(
        params['lstm']['title_cell'])(title_char_cnn_out)

    # title word
    title_wd_input = Input(
        shape=(params['title']['word']['sequence_length'],
               params['title']['word']['embedding_dim']),
        dtype='float32',
        name='title_wd_input')
    title_wd_cnn_out = keras_conv(
        title_wd_input,
        filters=params['CNN']['title_filters'],
        size_list=params['CNN']['title_sizes'],
        pooling_size=params['CNN']['pooling_size'],
        isMaxpool=params['CNN']['isMaxpool'])
    # LSTM cell
    title_wd_lstm_out = LSTM(params['lstm']['title_cell'])(title_wd_cnn_out)

    # desc char
    desc_char_input = Input(
        shape=(params['desc']['char']['sequence_length'],
               params['desc']['char']['embedding_dim']),
        dtype='float32',
        name='desc_char_input')
    desc_char_cnn_out = keras_conv(
        desc_char_input,
        filters=params['CNN']['desc_filters'],
        size_list=params['CNN']['desc_sizes'],
        pooling_size=params['CNN']['pooling_size'],
        isMaxpool=params['CNN']['isMaxpool'])
    # LSTM cell
    desc_char_lstm_out = LSTM(params['lstm']['desc_cell'])(desc_char_cnn_out)

    # desc word
    desc_word_input = Input(
        shape=(params['desc']['word']['sequence_length'],
               params['desc']['word']['embedding_dim']),
        dtype='float32',
        name='desc_word_input')
    desc_word_cnn_out = keras_conv(
        desc_word_input,
        filters=params['CNN']['desc_filters'],
        size_list=params['CNN']['desc_sizes'],
        pooling_size=params['CNN']['pooling_size'],
        isMaxpool=params['CNN']['isMaxpool'])
    # LSTM cell
    desc_word_lstm_out = LSTM(64)(desc_word_cnn_out)

    # Y0 predicted by title
    title_vec = concatenate(
        [title_wd_lstm_out, title_char_lstm_out], name='title_vec')
    # cells of combined lstm default by individual times two
    title_lstm_out = LSTM(
        int(params['lstm']['ratio'] * params['lstm']['title_cell']))(title_vec)

    Y0_output = Dense(
        params['Y0']['dim'], activation=params['Y0']['activation'],
        name='Y0')(title_lstm_out)

    # combine desc-char and desc-wd and put in LSTM cell
    desc_vec = concatenate(
        [desc_char_lstm_out, desc_word_lstm_out], name='desc_vec')
    # cells of combined lstm default by individual times two
    desc_lstm_out = LSTM(
        int(params['lstm']['ratio'] * params['lstm']['desc_cell']))(desc_vec)

    # assume label level information as abstract feature
    total_vec = concatenate([Y0_output, desc_lstm_out])
    total_lstm_out = LSTM(params['lstm']['deep_lstm_cell'])(
        total_vec) if params['lstm']['is_deep_lstm'] == 1 else total_vec

    Y1_output = Dense(
        params['Y1']['dim'], activation=params['Y1']['activation'],
        name='Y1')(total_lstm_out)

    model = Model(
        inputs=[
            title_char_input, title_wd_input, desc_char_input, desc_word_input
        ],
        outputs=[Y0_output, Y1_output])
    model.compile(
        optimizer=params['iter']['optimizer'],
        loss={
            'Y0_output': params['Y0']['loss'],
            'Y1_output': params['Y1']['loss']
        },
        loss_weights={
            'Y0_output': params['Y0']['loss_weight'],
            'Y1_output': params['Y1']['loss_weight']
        })
    return model


def keras_conv(X, filters, size_list, pooling_size=2, isMaxpool=True):
    """Redefine conv func."""
    pooled_output = []
    for size in size_list:
        conv = Conv1D(
            filters=filters,
            kernel_size=size,
            padding='valid',
            activation='relu',
            strides=1,
            bias_regularizer=l2(0.01), )(X)
        pooling = MaxPool1D(
            pool_size=pooling_size)(conv) if isMaxpool else AvgPool1D(
                pool_size=pooling_size)(conv)
        flatten = Flatten()(pooling)
        pooled_output.append(flatten)
    # combine all the pooled feature
    res = concatenate(
        pooled_output) if len(pooled_output) > 1 else pooled_output[0]
    return res


def assemble_base_lstm(params):
    """Construct flat lstm model by params."""
    # title word
    title_wd_input_shape = (
        params['title']['word']['sequence_length'],
        # params['title']['word']['embedding_dim']
    )
    # input
    X = Input(shape=title_wd_input_shape, dtype='int32', name='X')
    # embeding and filter padding value of index=0
    embedding = Embedding(
        output_dim=params['title']['word']['embedding_dim'],
        input_dim=params['title']['word']['vocab_size'],
        input_length=params['title']['word']['sequence_length'],
        name="embedding",
        mask_zero=True)(X)
    # filter mask_value
    # title_wd_input = Masking(
    #     mask_value=params['iter']['mask_value'],
    #     # input_shape=title_wd_input_shape,
    #     # dtype='float32',
    #     name='title_wd_input')(X)
    # batch norm
    # input_batch_norm = BatchNormalization()(embedding)
    # first layer
    title_wd_lstm_out = LSTM(params['lstm']['title_cell'], return_sequences=True)(embedding)
    title_wd_lstm_out = BatchNormalization(momentum=0.9)(title_wd_lstm_out)
    total_lstm_out = Dropout(0.3)(title_wd_lstm_out)
    # second layer
    title_wd_lstm_out = LSTM(params['lstm']['title_cell'])(title_wd_lstm_out)
    lstm_batch_norm = BatchNormalization(momentum=0.9)(title_wd_lstm_out)
    total_lstm_out = Dropout(0.3)(lstm_batch_norm)

    Y1 = Dense(params['Y1']['dim'])(total_lstm_out)
    Y1_output = Activation(params['Y1']['activation'], name='Y1')(BatchNormalization(momentum=0.9)(Y1))

    model = Model(inputs=X, outputs=Y1_output)
    model.compile(
        optimizer=params['iter']['optimizer'],
        loss=params['Y1']['loss'],
        metrics=[categorical_accuracy])
    return model
