# -*- coding: UTF-8 -*-
"""
    CBOW model for Java-AST-Node(Type) Embedding
    There are 92 types of Java-AST-Node...      [1, 92]
    ID 0 for Blank_Node, 92 + Empty_Node = 93   [0, 92]

    context_nodes -> target_node

    input: (parent_id, child1_id, child2_id, ...)
    predict: target_node

    context = sum(one_hot(parent_id, child1_id, child2_id, ...)) -> shape(93, )
    model:
        context : shape=(93, )
        embedding -> shape=(embedding_dim, )
        predict -> shape=(93, )
"""

import os
import sys
import numpy as np
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import InputLayer, Dense


def get_data(data_file):
    np_samples = np.load(data_file)
    np.random.shuffle(np_samples)
    x = contexts = np_samples[:, 0]
    y = targets = np_samples[:, 1]
    return x, y


def train_model(emb_dim, x, y, log_dir, model_file):
    # using default initializers

    input_layer = InputLayer(input_shape=(93,), name='input_layer')
    embedding_layer = Dense(units=emb_dim, name='embedding_layer', use_bias=False)
    out_layer = Dense(units=93, name='out_layer', activation='softmax')

    model = Sequential([input_layer, embedding_layer, out_layer])

    # print(model.get_config()) # out: dict
    # model.summary()  # out: table

    model.compile(optimizer='nadam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x, y,
                        batch_size=1024,
                        epochs=1000,
                        verbose=1,
                        callbacks=callbacks(log_dir, model_file),
                        validation_split=0,
                        validation_data=None,
                        shuffle=True,
                        initial_epoch=0)

    return history  # print(history.history)


def callbacks(log_dir, model_file):
    return [TensorBoard(log_dir=log_dir,
                        # embeddings_freq=1,
                        # embeddings_layer_names='embedding_layer',
                        # embeddings_metadata='metadata.tsv'
                        ),
            ModelCheckpoint(model_file,
                            monitor='loss',
                            verbose=1,
                            save_best_only=True,
                            save_weights_only=False,
                            mode='min',
                            period=1),
            EarlyStopping(monitor='loss',
                          min_delta=0,
                          patience=0,
                          verbose=1,
                          mode='min')]


def save_embedding_matrix(model_file, embedding_matrix_file_path):
    """save embedding_matrix as a single file"""
    embedding_matrix = load_model(model_file).get_layer('embedding_layer').get_weights()[0]
    embedding_matrix[0] = 0  # id 0 for Empty Node, let it=(0, 0, 0, ..., 0)
    np.save(embedding_matrix_file_path, embedding_matrix)


# -----------------------------------------------------------

emb_dim = int(sys.argv[1])

log_dir = os.path.join('logs', 'cbow', str(emb_dim))
model_file_path = os.path.join(log_dir, 'min_loss_model.hdf5')
embedding_matrix_file_path = os.path.join(log_dir, 'embedding_matrix_20.npy')

x, y = get_data('emb_samples.npy')

train_model(emb_dim, x, y, log_dir, model_file_path)

save_embedding_matrix(model_file_path, embedding_matrix_file_path)

