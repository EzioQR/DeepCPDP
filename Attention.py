import os
import json
import math
import random
from abc import abstractmethod
import numpy as np
import datetime

import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Layer, Embedding, Dense, Bidirectional, CuDNNLSTM, concatenate
from keras.callbacks import Callback, TensorBoard
from keras.utils import Sequence

from sklearn.metrics import f1_score, roc_curve, auc


# -----------------------------------------------

#os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def get_train_set(project_name, train_set_id):
    with open(os.path.join('dp_samples', 'samples.train.{}.json'.format(train_set_id)), 'r') as f:
        samples = json.load(f)
    return samples[project_name]


def get_test_set(project_name):
    with open(os.path.join('dp_samples', 'samples.test.json'), 'r') as f:
        samples = json.load(f)
    return samples[project_name]


def split_train_val(samples, val_percent=0.2):
    # split train:val= (1-val_percent):val_percent
    # stratified sampling

    # random.shuffle(samples)
    buggy_samples = [s for s in samples if s['label'] == 1]
    clean_samples = [s for s in samples if s['label'] == 0]

    p = round(len(samples) / 2 * (1 - val_percent))
    train_buggy, val_buggy = buggy_samples[:p], buggy_samples[p:]
    train_clean, val_clean = clean_samples[:p], clean_samples[p:]

    train = train_buggy + train_clean
    val = val_buggy + val_clean

    random.shuffle(train)
    random.shuffle(val)

    return train, val


class InputWrapper(Sequence):
    def __init__(self, samples, batch_size=4):
        self.samples = samples
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.samples) / self.batch_size)

    def __getitem__(self, idx):
        batch_samples = self.samples[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_x, batch_y = self._batch_to_ndarray(batch_samples)
        return batch_x, batch_y

    def on_epoch_end(self):
        pass

    @abstractmethod
    def _batch_to_ndarray(self, batch_sample):
        raise NotImplementedError


class TokenSequenceWrapper(InputWrapper):
    def _batch_to_ndarray(self, batch_sample):
        max_len_sequence = max(len(s['token_sequence']) for s in batch_sample)
        batch_sequence = [s['token_sequence'] + [0] * (max_len_sequence - len(s['token_sequence']))
                          for s in batch_sample]
        batch_label = [s['label'] for s in batch_sample]

        return np.asarray(batch_sequence, dtype=np.int), np.asarray(batch_label, dtype=np.int)


class TokenSequenceAndSoftwareMetricsWrapper(InputWrapper):
    def _batch_to_ndarray(self, batch_sample):
        max_len_sequence = max(len(s['token_sequence']) for s in batch_sample)
        batch_sequence = [s['token_sequence'] + [0] * (max_len_sequence - len(s['token_sequence']))
                          for s in batch_sample]

        batch_metrics = [s['metrics'] for s in batch_sample]

        batch_label = [s['label'] for s in batch_sample]

        return [np.asarray(batch_sequence), np.asarray(batch_metrics)], np.asarray(batch_label)


def compute_f1(model, wrapped_samples):
    y_true = np.asarray([s['label'] for s in wrapped_samples.samples])
    y_pred = np.round(model.predict_generator(wrapped_samples, steps=len(wrapped_samples), use_multiprocessing=True))
    return f1_score(y_true, y_pred)


def compute_auc(model, wrapped_samples):
    y_true = np.asarray([s['label'] for s in wrapped_samples.samples])
    y_pred = np.round(model.predict_generator(wrapped_samples, steps=len(wrapped_samples), use_multiprocessing=True))
    return auc(*(roc_curve(y_true, y_pred, pos_label=1)[0:2]))

def compute_all_pred(model, wrapped_samples):
    metrics_label = [(list(s["metrics"]) + [s['label']]) for s in wrapped_samples.samples]
    pred = model.predict_generator(wrapped_samples, steps=len(wrapped_samples), use_multiprocessing=True)
    pred_round = list(np.round(pred))
    pred = list(pred)
    res = []
    for i in range(len(metrics_label)):
        res.append(metrics_label[i] + [pred_round[i], pred[i]])
    return res
    

class CustomCallback(Callback):
    def __init__(self, wrapped_train_samples=None, wrapped_val_samples=None, score_calculator=compute_auc,
                 log_dir='temp_logs', patience=10, verbose=0):
        super(CustomCallback, self).__init__()
        #
        self.wrapped_train_samples = wrapped_train_samples
        self.wrapped_val_samples = wrapped_val_samples
        assert self.wrapped_val_samples is not None
        #
        self.score_calculator = score_calculator
        self.records = {'train_score': [], 'val_score': []}
        self.cur_max_val_score = 0
        self.best_epoch = 0

        #
        self.max_wait = patience
        #
        self.verbose = verbose
        #
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.best_weights_path = os.path.join(self.log_dir, 'weights.max_val_score.hdf5')

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        train_score = None
        if self.wrapped_train_samples is not None:
            train_score = self.score_calculator(self.model, self.wrapped_train_samples)
            self.records['train_score'].append(train_score)

        val_score = self.score_calculator(self.model, self.wrapped_val_samples)
        self.records['val_score'].append(val_score)
        #
        if self.verbose == 1: print("Epoch={}\ttrain_score={}\tval_score={}".format(epoch, train_score, val_score))

        # ---- early stopping ----
        if val_score >= self.cur_max_val_score:  # improving
            self.cur_max_val_score = val_score
            self.cur_wait = 0
            self.best_epoch = epoch
            self.model.save_weights(self.best_weights_path)
        else:  # not improving
            self.cur_wait += 1
            if self.cur_wait >= self.max_wait: self.model.stop_training = True

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        self.model.load_weights(self.best_weights_path)


# -----------------------------------------------

# embedding_matrix learned by ASTToken2Vec
def embedding_layer(trainable=False, rg=None):
    embedding_matrix_file_path = os.path.join('embedding_matrix_20.npy')
    embedding_matrix = np.load(embedding_matrix_file_path)
    return Embedding(input_dim=93, output_dim=20, embeddings_regularizer=rg, trainable=trainable,
                     weights=[embedding_matrix])


class AttentionWeightedAverage(Layer):
    """
    https://github.com/bfelbo/DeepMoji/blob/master/deepmoji/attlayer.py
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, regularizer=None, **kwargs):
        self.init = keras.initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        self.regularizer = regularizer
        super(AttentionWeightedAverage, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'return_attention': self.return_attention,
        }
        base_config = super(AttentionWeightedAverage, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.input_spec = [keras.engine.InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init,
                                 regularizer=keras.regularizers.get(self.regularizer))
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None

def random_blstm_attention_model(lstm_dim=512, lstm_regularizer='l2', merge_mode='ave',
                                       att_regularizer='l2',
                                       fc_dim=None, fc_regularizer=None,
                                       lg_regularizer='l2'):
    seq_len = None  # no limitation
    seq_id = Input(shape=[seq_len], dtype='int32', name='token_seq')
    #
    seq_vector = Embedding(input_dim=93, output_dim=20, trainable=True)(seq_id)
    #
    seq_feature = Bidirectional(
        CuDNNLSTM(units=lstm_dim, return_sequences=True,
                  kernel_regularizer=lstm_regularizer,
                  recurrent_regularizer=lstm_regularizer),
        merge_mode=merge_mode)(seq_vector)

    seq_feature = AttentionWeightedAverage(regularizer=att_regularizer, name='attention')(
        concatenate([seq_vector, seq_feature]))

    # seq_feature = AttentionWeightedAverage(regularizer=att_regularizer, name='attention')(seq_feature)

    if fc_dim is not None:
        seq_feature = Dense(fc_dim, activation='relu', kernel_regularizer=fc_regularizer)(seq_feature)

    output = Dense(1, activation='sigmoid', kernel_regularizer=lg_regularizer)(seq_feature)
    return Model(inputs=seq_id, outputs=output)

def asttoken2vec_blstm_attention_model(lstm_dim=512, lstm_regularizer='l2', merge_mode='ave',
                                       att_regularizer='l2',
                                       fc_dim=None, fc_regularizer=None,
                                       lg_regularizer='l2'):
    seq_len = None  # no limitation
    seq_id = Input(shape=[seq_len], dtype='int32', name='token_seq')
    #
    seq_vector = embedding_layer(trainable=False)(seq_id)  # id(1) -> one_hot(93) -> 20
    #
    seq_feature = Bidirectional(
        CuDNNLSTM(units=lstm_dim, return_sequences=True,
                  kernel_regularizer=lstm_regularizer,
                  recurrent_regularizer=lstm_regularizer),
        merge_mode=merge_mode)(seq_vector)

    seq_feature = AttentionWeightedAverage(regularizer=att_regularizer, name='attention')(
        concatenate([seq_vector, seq_feature]))

    # seq_feature = AttentionWeightedAverage(regularizer=att_regularizer, name='attention')(seq_feature)

    if fc_dim is not None:
        seq_feature = Dense(fc_dim, activation='relu', kernel_regularizer=fc_regularizer)(seq_feature)

    output = Dense(1, activation='sigmoid', kernel_regularizer=lg_regularizer)(seq_feature)
    return Model(inputs=seq_id, outputs=output)


# -------------------------------------------------------------------------------
folder = "asttoken2vec_blstm_attention_model"
if not os.path.isdir(folder): os.mkdir(folder)

file = open(os.path.join(folder, "output.csv"),"w")
file.write("source,target,0,1,2,3,4,5,6,7,8,9,avg\n")

model_creator, input_wrapper, score_calculator = [
    asttoken2vec_blstm_attention_model,
    TokenSequenceWrapper,
    compute_auc
]
results = []

batch_size = 64
patience = 10
log_dir = 'logs/test'
verbose = 1

source_projects = ['camel', 'poi', 'xalan', 'xerces']
target_projects = ['ant', 'camel', 'jedit', 'log4j', 'lucene', 'poi', 'synapse', 'velocity', 'xalan', 'xerces']

for i in range(10):
    print(i)
    file.write("{}\n".format(i))
    for source in source_projects:
        folder2 = os.path.join(folder, str(i))
        if not os.path.isdir(folder2): os.mkdir(folder2)
        time_file = open(os.path.join(folder2, "%s.txt" % source), "w")
        time_file.write("time1: {}\n".format(datetime.datetime.now())) ###

        train_samples, val_samples = split_train_val(get_train_set(source, i))

        train_samples = input_wrapper(train_samples, batch_size=batch_size)
        val_samples = input_wrapper(val_samples, batch_size=batch_size)

        callback = CustomCallback(wrapped_train_samples=None,
                                  wrapped_val_samples=val_samples,
                                  score_calculator=score_calculator,
                                  log_dir=log_dir,
                                  patience=patience,
                                  verbose=verbose)

        model = model_creator()

        time_file.write("time2: {}\n".format(datetime.datetime.now())) ###

        model.compile(loss='binary_crossentropy', optimizer='adam')

        time_file.write("time3: {}\n".format(datetime.datetime.now())) ###
        # if verbose == 1:  model.summary()
        model.fit_generator(generator=train_samples,
                            steps_per_epoch=len(train_samples),
                            shuffle=True,
                            epochs=100,
                            callbacks=[callback],  # TensorBoard(log_dir)
                            use_multiprocessing=True,
                            verbose=verbose)

        time_file.write("time4: {}\n".format(datetime.datetime.now())) ###

        for target in target_projects:
            if target == source: continue
            test_samples = input_wrapper(get_test_set(target), batch_size=batch_size)
            test_score = score_calculator(model, test_samples)
            print('{}'.format(test_score))
            file.write("{},{},{}\n".format(source, target, test_score))
            results.append((source, target, callback.best_epoch, max(callback.records['val_score']), test_score))
            
            with open(os.path.join(folder2, "%s-%s.txt" % (source, target)), "w") as t_v_file:
                data = compute_all_pred(model, test_samples)
                for column in data:
                    s = ""
                    for e in column:
                        s += "{},".format(e)
                    t_v_file.write(s[:-1] + "\n")
        
        time_file.write("time5: {}\n".format(datetime.datetime.now())) ###
        time_file.write("done.\n")
        time_file.close()

file.write("done.\n")
file.close()
# print()
# for i in range(len(results)):
#     r = results[i]
#
#     if i % 36 == 0: print(i / 36)
#
#     print('{}'.format(r[4]))
