import os
import time
import gc
import logging
import pickle

import numpy as np
from player.parse import parse_winner, parse_replay_file
from player.ndlstm import separable_lstm2
from player.state import GameState
from player.constants import MAX_BOARD_SIZE, FEATURE_SIZE, OUTPUT_SIZE, MOVE_TO_DIRECTION, OUTPUT_TO_MOVE, MOVE_TO_OUTPUT

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

class HaliteModel:

    def __init__(self, cached_model=None):
        self.h = MAX_BOARD_SIZE
        self.w = MAX_BOARD_SIZE
        self.hidden_size = 4
        self.learning_rate = 0.001
        self.batch_size = 10

        logging.warning("creating session")
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session()

            logging.warning("creating placeholders")
            # Build network
            # Input placeholder
            self.x = tf.placeholder(tf.float32, [None, self.h, self.w, FEATURE_SIZE])
            # Output placeholder
            self.y = tf.placeholder(tf.int32, [None])
            self.weights = tf.placeholder(tf.float32, [OUTPUT_SIZE])
            self.mask = tf.placeholder(tf.int32, [None, 3])
            self.seq_lengths1 = tf.placeholder(tf.int32, [None])
            self.seq_lengths2 = tf.placeholder(tf.int32, [None])
            self.training = tf.placeholder(tf.bool)

            he_init = tf.contrib.layers.variance_scaling_initializer()
            bn_params = {
                'training': self.training,
            }

            logging.warning("creating graph")
            start = time.time()
            normalized1 = tf.layers.batch_normalization(self.x, training=self.training)
            rnn1 = separable_lstm2(GRUCell, self.hidden_size, normalized1, self.seq_lengths1, self.seq_lengths2)
            #normalized2 = tf.layers.batch_normalization(rnn1, training=self.training)
            #rnn2 = separable_lstm2(GRUCell, self.hidden_size, rnn1, self.seq_lengths)
            #normalized3 = tf.layers.batch_normalization(rnn2, training=self.training)
            #rnn3 = separable_lstm2(GRUCell, self.hidden_size, rnn2, self.seq_lengths)
            print(rnn1)
            relu1 = slim.fully_connected(inputs=rnn1, num_outputs=self.hidden_size,
                weights_initializer=he_init, normalizer_fn=tf.layers.batch_normalization, normalizer_params=bn_params)
            relu2 = slim.fully_connected(inputs=relu1, num_outputs=self.hidden_size,
                weights_initializer=he_init, normalizer_fn=tf.layers.batch_normalization, normalizer_params=bn_params)
            relu3 = slim.fully_connected(inputs=relu2, num_outputs=self.hidden_size,
                weights_initializer=he_init, normalizer_fn=tf.layers.batch_normalization, normalizer_params=bn_params)

            self.logits = slim.fully_connected(inputs=relu3,
                                             num_outputs=OUTPUT_SIZE,
                                             activation_fn=None)
            self.predictions = tf.nn.top_k(self.logits)
            weighted_logits = tf.multiply(self.logits, tf.broadcast_to(self.weights, tf.shape(self.logits)))
            self.xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=weighted_logits)
            self.loss = tf.reduce_mean(self.xentropy)
            self.grad_update = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.logits, self.y, 2), tf.float32))
            self.saver = tf.train.Saver()

            if cached_model is None:
                self.session.run(tf.global_variables_initializer())
            else:
                logging.warning('restoring model')
                self.saver.restore(self.session, cached_model)


    def predict_moves(self, game_map, me, other_players, turn_number):
        my_ships = {s.id: s for s in me.get_ships()}
        opp_ships = {s.id: s for p in other_players for s in p.get_ships()}
        my_dropoffs = list(me.get_dropoffs()) + [me.shipyard]
        opp_dropoffs = [d for p in other_players for d in p.get_dropoffs()] + \
                       [p.shipyard for p in other_players]
        frame = [[y.halite_amount for y in x] for x in game_map._cells]
        game_state = GameState(turn_number, frame, {}, my_ships, opp_ships, my_dropoffs, opp_dropoffs)
        feature_map = np.expand_dims(game_state.get_feature_map(), axis=0)
        ship_mask = np.expand_dims(game_state.get_ship_mask(), axis=0)
        mask_idx = np.stack(list(np.nonzero(ship_mask)), axis=1)

        sequence_lengths = [ MAX_BOARD_SIZE for i in range(MAX_BOARD_SIZE)]
        feed_dict = {self.x: feature_map, self.mask: mask_idx, self.seq_lengths: sequence_lengths, self.training: False}
        _, moves = self.session.run([self.predictions], feed_dict=feed_dict)[0]
        move_dict = {}
        for k, v in my_ships.items():
            move_label = moves[0][v.position.x][v.position.y][0]
            move_dict[k] = MOVE_TO_DIRECTION[OUTPUT_TO_MOVE[move_label]]

        return move_dict

    def test_eval(self):
        batch = next(data_gen('../test', 400))
        training_predictions, accuracy = self.run_batch(
            [self.predictions, self.accuracy], batch, False)
        print('Test Accuracy: {}'.format(accuracy))
        print(batch[1])
        print(np.ndarray.flatten(training_predictions[1]))

    def run_batch(self, eval_list, batch, training):
        weights = np.array([0.1,1,1,1,1])
        size_list, moves, features = batch
        sequence_lengths1 = []
        for map_size in size_list:
            sequence_lengths1 += [ map_size for i in range(MAX_BOARD_SIZE)]
        sequence_lengths2 = np.array(size_list)
        feed_dict = {self.x: features, self.y:moves,
                     self.seq_lengths1: sequence_lengths1, self.seq_lengths2: sequence_lengths2, self.training: True, self.weights: weights}
        return self.session.run(eval_list, feed_dict=feed_dict)

    def train_on_files(self, folder, ckpt_file):
        #weights = generate_weights(train[:10])

        map_size = 0
        i = 0
        for batch in data_gen('../train', 20):
            t = time.time()
            training_predictions, loss, _, accuracy = self.run_batch(
                [self.predictions, self.loss, self.grad_update, self.accuracy], batch, True)
            #print('--- {} ---'.format(time.time() - t))
            #print(batch[1])
            #print(np.ndarray.flatten(training_predictions[1]))
            #print('Accuracy: {}'.format(accuracy))
            i += 1
            if not i % 50:
                self.test_eval()
