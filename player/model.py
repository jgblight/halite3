import os
import glob
import time
import gc
import logging
import pickle

import numpy as np
from player.parse import parse_winner, parse_replay_file
from player.ndlstm import separable_lstm2
from player.state import GameState
from player.constants import MAX_BOARD_SIZE, FEATURE_SIZE, OUTPUT_SIZE, MOVE_TO_DIRECTION, OUTPUT_TO_MOVE, MOVE_TO_OUTPUT
from player.utils import Timer

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.contrib.rnn import GRUCell

def train_test_split(folder, data_size, split=0.2):
    files = np.array(sorted([ os.path.join(folder, f) for f in os.listdir(folder)]))
    indices = np.random.permutation(files.shape[0])
    test_size = int(data_size*split)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:data_size]
    return files[train_indices], files[test_indices]

def randomize_order(folder):
    files = np.array(glob.glob(os.path.join(folder, '*.pkl')))
    random_files = np.random.permutation(files)


def data_gen(folder, batch_size):
    files = np.array(sorted([ os.path.join(folder, f) for f in os.listdir(folder)]))
    random_files = np.random.permutation(files)
    for start in range(0, len(random_files), batch_size):
        end = min(start+batch_size, len(random_files))
        batch = random_files[start:end]
        feature_list = []
        move_list = []
        size_list = []
        for filename in batch:
            with open(filename, 'rb') as f:
                map_size, move, features = pickle.load(f)
            size_list.append(map_size)
            feature_list.append(features)
            move_list.append(MOVE_TO_OUTPUT[move])
        feature_arr = np.stack(feature_list, axis=0)
        yield size_list, np.array(move_list), feature_arr


def one_hot(arr, depth):
    arr_len = arr.shape[0]
    oneh = np.zeros((arr_len,depth))
    oneh[np.arange(arr_len), arr.astype(int)] = 1
    return oneh

def generate_weights(training_files):
    counts = np.zeros((OUTPUT_SIZE,))
    for i, training_file in enumerate(training_files):
        for data in data_gen(training_file, 100):
            _, _, mask, moves = data
            move_logits = moves[mask[:,0], mask[:,1], mask[:,2]]
            weights = np.sum(one_hot(move_logits, OUTPUT_SIZE),axis=0)
            counts += weights
    counts = np.sum(counts) / counts
    return counts

def new_conv_layer(input, num_input_channels, filter_size, num_filters, name):

    with tf.variable_scope(name) as scope:
        # Shape of the filter-weights for the convolution
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        # Create new weights (filters) with the given shape
        weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))

        # Create new biases, one for each filter
        biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))

        # TensorFlow operation for convolution
        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

        # Add the biases to the results of the convolution.
        layer += biases

        return layer, weights

def new_pool_layer(input, name):

    with tf.variable_scope(name) as scope:
        # TensorFlow operation for convolution
        layer = tf.nn.max_pool(value=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        return layer

def new_relu_layer(input, name):

    with tf.variable_scope(name) as scope:
        # TensorFlow operation for convolution
        layer = tf.nn.relu(input)

        return layer

class HaliteModel:

    def __init__(self, cached_model=None):
        self.h = MAX_BOARD_SIZE
        self.w = MAX_BOARD_SIZE
        self.hidden_size = 50
        self.learning_rate = 0.001
        self.batch_size = 10

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session()

            # Build network
            # Input placeholder
            self.x = tf.placeholder(tf.float32, [None, self.h, self.w, FEATURE_SIZE])
            #self.x = tf.placeholder(tf.float32, [None, 83])
            # Output placeholder
            self.y = tf.placeholder(tf.int32, [None])
            self.mask = tf.placeholder(tf.int32, [None, 3])
            self.training = tf.placeholder(tf.bool)

            conv1_fmaps = 30
            conv1_ksize = 3

            conv2_fmaps = 20
            conv2_ksize = 3

            conv3_fmaps = 20
            conv3_ksize = 5
            dropout_rate = 0.20

            he_init = tf.contrib.layers.variance_scaling_initializer()
            bn_params = {
                'training': self.training,
            }

            start = time.time()

            x_norm = tf.layers.batch_normalization(self.x, training=self.training)
            conv1, conv1_weights = new_conv_layer(x_norm, FEATURE_SIZE, conv1_ksize, conv1_fmaps, 'conv1')
            pool1 = new_pool_layer(conv1, 'pool1')
            relu1 = new_relu_layer(pool1, 'relu1')

            conv2, conv2_weights = new_conv_layer(relu1, conv1_fmaps, conv2_ksize, conv2_fmaps, 'conv2')
            pool2 = new_pool_layer(conv2, 'pool2')
            relu2 = new_relu_layer(pool2, 'relu2')

            conv3, conv3_weights = new_conv_layer(relu2, conv2_fmaps, conv3_ksize, conv3_fmaps, 'conv3')
            pool3 = new_pool_layer(conv3, 'pool3')
            relu3 = new_relu_layer(pool3, 'relu3')

            flat = tf.layers.flatten(relu3)

            fc1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=50,
                weights_initializer=he_init, normalizer_fn=tf.layers.batch_normalization, normalizer_params=bn_params)
            d1 = tf.layers.dropout(inputs=fc1, rate=dropout_rate, training=self.training)
            fc2 = tf.contrib.layers.fully_connected(inputs=d1, num_outputs=40,
                weights_initializer=he_init, normalizer_fn=tf.layers.batch_normalization, normalizer_params=bn_params)
            d2 = tf.layers.dropout(inputs=fc1, rate=dropout_rate, training=self.training)
            fc3 = tf.contrib.layers.fully_connected(inputs=d2, num_outputs=30,
                weights_initializer=he_init, normalizer_fn=tf.layers.batch_normalization, normalizer_params=bn_params)
            d3 = tf.layers.dropout(inputs=fc1, rate=dropout_rate, training=self.training)
            fc4 = tf.contrib.layers.fully_connected(inputs=d3, num_outputs=20,
                weights_initializer=he_init, normalizer_fn=tf.layers.batch_normalization, normalizer_params=bn_params)
            d4 = tf.layers.dropout(inputs=fc1, rate=dropout_rate, training=self.training)
            fc5 = tf.contrib.layers.fully_connected(inputs=d4, num_outputs=10,
                weights_initializer=he_init, normalizer_fn=tf.layers.batch_normalization, normalizer_params=bn_params)

            self.logits = slim.fully_connected(inputs=fc5,
                                             num_outputs=OUTPUT_SIZE,
                                             activation_fn=None)
            self.predictions = tf.nn.top_k(self.logits, 2)
            self.xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
            self.loss = tf.reduce_mean(self.xentropy)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.grad_update = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.logits, self.y, 2), tf.float32))
            self.saver = tf.train.Saver()

            if cached_model is None:
                self.session.run(tf.global_variables_initializer())
            else:
                self.saver.restore(self.session, cached_model)

    def predict_move(self, game_state, ship_id):
        feature_list = []
        feature_list.append(game_state.feature_shift(ship_id))
        feature_map = np.stack(feature_list, axis=0)

        with Timer("Generate Prediction"):
            feed_dict = {self.x: feature_map, self.training: False}
            predictions = self.session.run([self.predictions], feed_dict=feed_dict)[0]
        _, moves = predictions
        move_dict = {}
        moves = np.ndarray.flatten(moves)
        return [MOVE_TO_DIRECTION[OUTPUT_TO_MOVE[x]] for x in moves]

    def test_eval(self):
        batch = next(data_gen('../test', 1000))
        accuracy = self.run_batch([self.accuracy], batch, False)
        batch = next(data_gen('../test', 100))
        training_predictions = self.run_batch([self.predictions], batch, False)[0]
        print('Test Accuracy: {}'.format(accuracy))
        print(batch[1])
        print(logits)
        print(np.ndarray.flatten(training_predictions[1]))

    def run_batch(self, eval_list, batch, training):
        weights = np.array([0.1,1,1,1,1])
        size_list, moves, features = batch
        sequence_lengths1 = []
        for map_size in size_list:
            sequence_lengths1 += [ map_size for i in range(MAX_BOARD_SIZE)]
        sequence_lengths2 = np.array(size_list)
        feed_dict = {self.x: features, self.y:moves, self.training: training}
        return self.session.run(eval_list, feed_dict=feed_dict)

    def train_on_files(self, folder, ckpt_file):
        map_size = 0
        i = 0
        epochs = 10
        for x in range(epochs):
            for batch in data_gen(folder, 20):
                t = time.time()
                training_predictions, loss, _, accuracy = self.run_batch(
                    [self.predictions, self.loss, self.grad_update, self.accuracy], batch, True)
                #print('--- {} ---'.format(time.time() - t))
                #print(batch[1])
                #print(np.ndarray.flatten(training_predictions[1]))
                #print('Accuracy: {}'.format(accuracy))
                i += 1
                if not (i % 1000):
                    print("Training Loss: {}".format(loss))
                    self.test_eval()
                    self.saver.save(self.session, ckpt_file.format(i))
