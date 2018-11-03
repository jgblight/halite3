import os
import time
import gc
import logging

import numpy as np
from player.parse import parse_winner
from player.ndlstm import separable_lstm
from player.state import GameState
from player.constants import MAX_BOARD_SIZE, FEATURE_SIZE, OUTPUT_SIZE, MOVE_TO_DIRECTION, OUTPUT_TO_MOVE

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

def data_gen(filename, batch_size):
    states = parse_winner(filename)
    first_state = states[0]
    map_size = first_state.map_size
    for start_idx in range(0, len(states), batch_size):
        feature_list = []
        mask_list = []
        move_list = []
        end_idx = min(start_idx+batch_size, len(states))
        for turn_idx in range(start_idx, end_idx):
            s = states[turn_idx]
            feature_list.append(s.get_feature_map())
            mask_list.append(s.get_ship_mask())
            move_list.append(s.get_expected_moves())
        features = np.stack(feature_list, axis=0)
        masks = np.stack(mask_list, axis=0)
        moves = np.stack(move_list, axis=0)
        mask = np.stack(list(np.nonzero(masks)), axis=1)
        yield map_size, features, mask, moves

class HaliteModel:

    def __init__(self, cached_model=None):
        self.h = MAX_BOARD_SIZE
        self.w = MAX_BOARD_SIZE
        self.hidden_size = 7
        self.learning_rate = 0.001
        self.batch_size = 10

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._session = tf.Session()


            # Build network
            # Input placeholder
            self._x = tf.placeholder(tf.float32, [None, self.h, self.w, FEATURE_SIZE])
            # Output placeholder
            self._y = tf.placeholder(tf.int32, [None, self.h, self.w])
            self._mask = tf.placeholder(tf.int32, [None, 3])
            self._seq_lengths = tf.placeholder(tf.int32, [None])
            self._training = tf.placeholder(tf.bool)

            normalized = tf.layers.batch_normalization(self._x)
            rnn1 = separable_lstm(GRUCell, self.hidden_size, normalized, self._seq_lengths)

            model_out = slim.fully_connected(inputs=rnn1,
                                             num_outputs=OUTPUT_SIZE,
                                             activation_fn=None)
            self._predictions = tf.nn.top_k(model_out)
            logits = tf.gather_nd(model_out, self._mask)
            labels = tf.gather_nd(self._y, self._mask)
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            self._loss = tf.reduce_mean(xentropy)
            self._grad_update = tf.train.AdamOptimizer(self.learning_rate).minimize(self._loss)
            self._accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32))
            self._saver = tf.train.Saver()

            if cached_model is None:
                self._session.run(tf.global_variables_initializer())
            else:
                self._saver.restore(self._session, cached_model)


    def predict_moves(self, game_map, me, other_players, turn_number):
        my_ships = {s.id: s for s in me.get_ships()}
        opp_ships = {s.id: s for p in other_players for s in p.get_ships()}
        my_dropoffs = list(me.get_dropoffs()) + [me.shipyard]
        opp_dropoffs = [d for p in other_players for d in p.get_dropoffs()] + \
                       [p.shipyard for p in other_players]
        frame = [[y.halite_amount for y in x] for x in game_map._cells]
        game_state = GameState(frame, {}, my_ships, opp_ships, my_dropoffs, opp_dropoffs)
        feature_map = np.expand_dims(game_state.get_feature_map(), axis=0)
        ship_mask = np.expand_dims(game_state.get_ship_mask(), axis=0)
        mask_idx = np.stack(list(np.nonzero(ship_mask)), axis=1)

        sequence_lengths = [ MAX_BOARD_SIZE for i in range(MAX_BOARD_SIZE)]
        feed_dict = {self._x: feature_map, self._mask: mask_idx, self._seq_lengths: sequence_lengths, self._training: False}
        _, moves = self._session.run([self._predictions], feed_dict=feed_dict)[0]
        logging.warning(moves)
        move_dict = {}
        for k, v in my_ships.items():
            move_label = moves[0][v.position.x][v.position.y][0]
            move_dict[k] = MOVE_TO_DIRECTION[OUTPUT_TO_MOVE[move_label]]

        return move_dict

    def train_on_files(self, folder, ckpt_file):
        split = 0.2
        train, test = train_test_split(folder, 100, split)

        split_1 = int(1/split)
        samples = 5
        for i, training_file in enumerate(train):
            for data in data_gen(training_file, self.batch_size):
                map_size, features, mask_idx, moves = data
                sequence_lengths = [ map_size for i in range(MAX_BOARD_SIZE*features.shape[0])]
                feed_dict = {self._x: features, self._y:moves, self._mask: mask_idx,
                             self._seq_lengths: sequence_lengths, self._training: True}
                self._session.run([self._loss, self._grad_update], feed_dict=feed_dict)
            if not i % split_1:
                print('Training Accuracy: {}'.format(self._accuracy.eval(session=self._session, feed_dict=feed_dict)))
                test_file = test[int(i*0.2)]
                map_size, features, mask_idx, moves = next(data_gen(test_file, 400))
                sequence_lengths = [ map_size for i in range(MAX_BOARD_SIZE*features.shape[0])]
                feed_dict = {self._x: features, self._y:moves, self._mask: mask_idx,
                             self._seq_lengths: sequence_lengths, self._training: False}
                print('Test Accuracy: {}'.format(self._accuracy.eval(session=self._session, feed_dict=feed_dict)))
                print('Progress: {}%'.format(100*i/len(train)))
        self._saver.save(self._session, ckpt_file)
