import numpy as np
from utils.data_helper import train_word2vec, load_data
from keras.optimizers import Adagrad
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPool1D, Conv1D, Embedding,ActivityRegularization,AvgPool1D
from keras.layers.merge import Concatenate
from keras.preprocessing import sequence
from keras.layers.normalization import BatchNormalization

from copy import deepcopy
np.random.seed(0)

# ---------------------- Parameters section -------------------
#
# Model type. See Kim Yoon's Convolutional Neural Networks for Sentence
# Classification, Section 3
model_type = "CNN-rand"  # CNN-rand|CNN-non-static|CNN-static

# Model Hyperparameters
embedding_dim = 100
filter_sizes = (3, 4, 5)
num_filters = 256
dropout_prob = (0.2, 0.3)
hidden_dims = 128

# Training parameters
batch_size = 200
num_epochs = 10

# Prepossessing parameters
sequence_length = 23
num_class = 47

# Word2Vec parameters (see train_word2vec)
min_word_count = 1
context = 10
#
# ---------------------- Parameters end -----------------------


# Data Preparation
print("Load data...")
trn_text, trn_labels, tst_text, tst_labels, vocabulary, vocabulary_inv = load_data('../docs/CNN/mytest',
                                                                                   use_tst=True,
                                                                                   lbl_text_index=[
                                                                                       0, 1],
                                                                                   split_tag='@@@',
                                                                                   padding_mod='average',
                                                                                   is_shuffle=True,
                                                                                   ratio=0.2)

# Y0 = [y.strip('\n') for y in open('../docs/CNN/Y0').readlines()]
Y1 = [y.strip('\n') for y in open('../docs/CNN/Y1').readlines()]


res = np.zeros((len(trn_labels), len(Y1)))
for i, yy in enumerate(trn_labels):
    for lbl in yy:
        if lbl in Y1:
            res[i][Y1.index(lbl)] = 1
y_train = deepcopy(res)

res = np.zeros((len(tst_labels), len(Y1)))
for i, yy in enumerate(tst_labels):
    for lbl in yy:
        if lbl in Y1:
            res[i][Y1.index(lbl)] = 1
y_test = deepcopy(res)

print("x_train shape:", trn_text.shape)
print("x_test shape:", tst_text.shape)
print('y shape :', y_test.shape)

# Prepare embedding layer weights and convert inputs for static model
print("Model type is", model_type)
if model_type == "CNN-non-static" or model_type == "CNN-static":
    embedding_weights = train_word2vec(np.vstack((trn_text, tst_text)), vocabulary_inv, num_features=embedding_dim,
                                       min_word_count=min_word_count, context=context)
    if model_type == "CNN-static":
        x_train = embedding_weights[0][trn_text]
        x_test = embedding_weights[0][tst_text]
elif model_type == "CNN-rand":
    x_train = deepcopy(trn_text)
    x_test = deepcopy(tst_text)
    embedding_weights = None
else:
    raise ValueError("Unknown model type")


# Build model
input_shape = (sequence_length,
               embedding_dim) if model_type == "CNN-static" else (sequence_length,)
model_input = Input(shape=input_shape)

# Static model do not have embedding layer
if model_type == "CNN-static":
    z = Dropout(dropout_prob[0])(model_input)
else:
    z = Embedding(len(vocabulary_inv), embedding_dim,
                  input_length=sequence_length, name="embedding")(model_input)
    z = Dropout(dropout_prob[0])(z)

# Convolutional block
conv_blocks = []
for sz in filter_sizes:
    conv = Conv1D(filters=num_filters,
                  kernel_size=sz,
                  padding="same",
                  activation="relu",
                  strides=1)(z)
    conv = MaxPool1D(pool_size=2)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

# z = BatchNormalization(beta_initializer='ones')(z)
z = Dropout(dropout_prob[1])(z)
z = Dense(hidden_dims, activation="relu")(z)
# z = ActivityRegularization(l1=0.1)(z)

model_output = Dense(num_class, activation="softmax")(z)

model = Model(model_input, model_output)
model.compile(loss="categorical_crossentropy",
              optimizer=Adagrad(1e-2), metrics=["accuracy"])

# Initialize weights with word2vec
if model_type == "CNN-non-static":
    embedding_layer = model.get_layer("embedding")
    embedding_layer.set_weights(embedding_weights)

# Train the model
model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
          validation_data=(x_test, y_test), verbose=1)

probs = model.predict(x_test)
Y1 = np.array(Y1)
for i in range(200):
    print(' '.join([vocabulary_inv[x] for x in x_test[i]]))
    print('ground truth: %s'%Y1[y_test[i]==1][0])
    print('predict label: %s'%Y1[np.argmax(probs[i])])

print('evalution....')
total_n = y_test.shape[0]
tp = 0.
for i in range(total_n):
    if np.where(y_test[i]==1)[0][0] == np.argmax(probs[i]):
        tp += 1
print(tp/total_n)
