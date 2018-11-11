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
    #TODO: better batching
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
        for i in range(0, mask.shape[0], 5):
            if i+5 > mask.shape[0]:
                continue
            mask_slice = mask[i:i+5,:]
            yield map_size, features, mask_slice, moves

def data_gen2(filename, batch_size):
    feature_list = []
    idx_list = []
    move_list = []
    states = parse_winner(filename)
    first_state = states[0]
    map_size = first_state.map_size
    for i in np.random.permutation(len(states))[:20]:
        s = states[i]
        feature_map = s.get_feature_map()
        moves = s.get_ship_moves()
        for j in range(len(moves)):
            move_tuple = moves[j]
            new_idx = [len(feature_list), move_tuple[1][0], move_tuple[1][1]]
            feature_list.append(feature_map)
            idx_list.append(new_idx)
            move_list.append(move_tuple[0])
            if len(feature_list) >= batch_size:
                f = np.stack(feature_list, axis=0)
                mi = np.stack(idx_list, axis=0)
                mo = np.stack(move_list, axis=0)
                yield map_size, f, mi, mo
                feature_list = []
                idx_list = []
                move_list = []
    if len(feature_list):
        f = np.stack(feature_list, axis=0)
        mi = np.stack(idx_list, axis=0)
        mo = np.stack(move_list, axis=0)
        yield map_size, f, mi, mo

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
        self.hidden_size = 12
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
            self.seq_lengths = tf.placeholder(tf.int32, [None])
            self.training = tf.placeholder(tf.bool)

            he_init = tf.contrib.layers.variance_scaling_initializer()
            bn_params = {
                'training': self.training,
            }

            logging.warning("creating graph")
            start = time.time()
            normalized1 = tf.layers.batch_normalization(self.x, training=self.training)
            rnn1 = separable_lstm(GRUCell, self.hidden_size, normalized1, self.seq_lengths)
            #normalized2 = tf.layers.batch_normalization(rnn1, training=self.training)
            rnn2 = separable_lstm(GRUCell, self.hidden_size, rnn1, self.seq_lengths)
            #normalized3 = tf.layers.batch_normalization(rnn2, training=self.training)
            rnn3 = separable_lstm(GRUCell, self.hidden_size, rnn2, self.seq_lengths)
            gathered =tf.gather_nd(rnn3, self.mask)
            relu1 = slim.fully_connected(inputs=gathered, num_outputs=self.hidden_size,
                weights_initializer=he_init, normalizer_fn=tf.layers.batch_normalization, normalizer_params=bn_params)
            relu2 = slim.fully_connected(inputs=relu1, num_outputs=self.hidden_size,
                weights_initializer=he_init, normalizer_fn=tf.layers.batch_normalization, normalizer_params=bn_params)
            relu3 = slim.fully_connected(inputs=relu2, num_outputs=self.hidden_size,
                weights_initializer=he_init, normalizer_fn=tf.layers.batch_normalization, normalizer_params=bn_params)
            relu4 = slim.fully_connected(inputs=relu3, num_outputs=self.hidden_size,
                weights_initializer=he_init, normalizer_fn=tf.layers.batch_normalization, normalizer_params=bn_params)
            relu5 = slim.fully_connected(inputs=relu4, num_outputs=self.hidden_size,
                weights_initializer=he_init, normalizer_fn=tf.layers.batch_normalization, normalizer_params=bn_params)

            self.logits = slim.fully_connected(inputs=relu5,
                                             num_outputs=OUTPUT_SIZE,
                                             activation_fn=None)
            self.predictions = tf.nn.top_k(self.logits)
            self.training_predictions = tf.nn.top_k(self.logits)
            weighted_logits = tf.multiply(self.logits, tf.broadcast_to(self.weights, tf.shape(self.logits)))
            self.xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=weighted_logits)
            self.loss = tf.reduce_mean(self.xentropy)
            self.grad_update = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.logits, self.y, 1), tf.float32))
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

    def test_eval(self, test_file):
        map_size, features, mask_idx, moves = next(data_gen2(test_file, 400))
        sequence_lengths = [ map_size for i in range(MAX_BOARD_SIZE*features.shape[0])]
        feed_dict = {self.x: features, self.y:moves, self.mask: mask_idx,
                     self.seq_lengths: sequence_lengths, self.training: False}
        training_predictions, accuracy = self.session.run([self.training_predictions, self.accuracy], feed_dict=feed_dict)
        print('Test Accuracy: {}'.format(accuracy))
        print(moves)
        print(np.ndarray.flatten(training_predictions[1]))

    def train_on_files(self, folder, ckpt_file):
        split = 0.2
        train, test = train_test_split(folder, 200, split)

        split_1 = int(1/split)
        samples = 5
        #weights = generate_weights(train[:10])
        weights = np.array([1,1,1,1,0.1])

        self.test_eval(test[0])
        for i, training_file in enumerate(train):
            for data in data_gen2(training_file, 16):
                t = time.time()
                map_size, features, mask_idx, moves = data
                sequence_lengths = [ map_size for i in range(MAX_BOARD_SIZE*features.shape[0])]
                feed_dict = {self.x: features, self.y:moves, self.mask: mask_idx,
                             self.seq_lengths: sequence_lengths, self.training: True, self.weights: weights}
                training_predictions, loss, _ = self.session.run([self.training_predictions, self.loss, self.grad_update], feed_dict=feed_dict)
                #print('--- {} ---'.format(time.time() - t))
                #print(moves)
                #print(np.ndarray.flatten(training_predictions[1]))
            if not i%10:
                print('Training Accuracy: {}'.format(self.accuracy.eval(session=self.session, feed_dict=feed_dict)))
                test_file = test[int(i*0.2)]
                self.test_eval(test_file)
                print('Progress: {}%'.format(100*i/len(train)))
                self.saver.save(self.session, ckpt_file.format(i))
