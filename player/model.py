import os
import time
import pickle
import json
import random
import string

from player.utils import Timer, log_message

with Timer("halite import", True):
    from player.state import GameState
    from player.constants import MAX_BOARD_SIZE, FEATURE_SIZE, OUTPUT_SIZE, MOVE_TO_DIRECTION, OUTPUT_TO_MOVE, MOVE_TO_OUTPUT

with Timer("numpy import", True):
    import numpy as np

with Timer("tf import", True):
    import tensorflow as tf

from player.tf_contrib_copy import fully_connected, variance_scaling_initializer

#with Timer("slim import", True):
#    import tensorflow.contrib.slim as slim

def train_test_split(folder, data_size, split=0.2):
    files = np.array(sorted([ os.path.join(folder, f) for f in os.listdir(folder)]))
    indices = np.random.permutation(files.shape[0])
    test_size = int(data_size*split)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:data_size]
    return files[train_indices], files[test_indices]

def data_gen(folder, batch_size, parse_file):
    files = np.array(sorted([ os.path.join(folder, f) for f in os.listdir(folder)]))
    random_files = np.random.permutation(files)
    for start in range(0, len(random_files), batch_size):
        end = min(start+batch_size, len(random_files))
        batch = random_files[start:end]
        feature_list = []
        move_list = []
        for filename in batch:
            with open(filename, 'rb') as f:
                move, features = parse_file(f)
            feature_list.append(features)
            move_list.append(move)
        feature_arr = np.stack(feature_list, axis=0)
        yield np.array(move_list), feature_arr

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

class ModelParams:

    def __init__(self, name, params):
        self.name = name
        self.params = params

    @staticmethod
    def new():
        name = ''.join([random.choice(string.ascii_lowercase) for x in range(5)])
        return ModelParams(name, {})

    @staticmethod
    def from_json(filename):
        f = open(filename)
        data = json.load(f)
        return ModelParams(data['name'], data['params'])

    def to_json(self, filename, extra):
        data = {'name': self.name, 'params': self.params}
        data.update(extra)
        f = open(filename, 'w')
        json.dump(data, f)
        f.close()

    def get_int(self, key, lower, upper):
        if key not in self.params:
            value = int(np.random.random_integers(lower, upper))
            self.params[key] = value
        return self.params[key]


class HaliteModel:

    def __init__(self, categories, cached_model=None, params_file=None, train_folder=None, test_folder=None):
        self.learning_rate = 0.001
        self.batch_size = 20
        self.categories = categories
        self.train_folder = train_folder
        self.test_folder = test_folder


        if params_file:
            self.params = ModelParams.from_json(params_file)
        else:
            self.params = ModelParams.new()

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session()

            # Build network
            # Input placeholder
            self.x = tf.placeholder(tf.float32, [None, MAX_BOARD_SIZE, MAX_BOARD_SIZE, FEATURE_SIZE])
            #self.x = tf.placeholder(tf.float32, [None, 83])
            # Output placeholder
            self.y = tf.placeholder(tf.int32, [None])
            self.training = tf.placeholder(tf.bool)

            conv1_fmaps = self.params.get_int('conv1_fmaps', 20, 50)
            conv1_ksize = 3

            conv2_fmaps = self.params.get_int('conv2_fmaps', 20, 40)
            conv2_ksize = 3

            conv3_fmaps = self.params.get_int('conv3_fmaps', 20, 40)
            conv3_ksize = 5
            dropout_rate = 0.20

            he_init = variance_scaling_initializer()
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

            fc1 = fully_connected(inputs=flat, num_outputs=self.params.get_int('fc1', 30, 60),
                weights_initializer=he_init, normalizer_fn=tf.layers.batch_normalization, normalizer_params=bn_params)
            d1 = tf.layers.dropout(inputs=fc1, rate=dropout_rate, training=self.training)
            fc2 = fully_connected(inputs=d1, num_outputs=self.params.get_int('fc2', 30, 60),
                weights_initializer=he_init, normalizer_fn=tf.layers.batch_normalization, normalizer_params=bn_params)
            d2 = tf.layers.dropout(inputs=fc1, rate=dropout_rate, training=self.training)
            fc3 = fully_connected(inputs=d2, num_outputs=self.params.get_int('fc3', 20, 50),
                weights_initializer=he_init, normalizer_fn=tf.layers.batch_normalization, normalizer_params=bn_params)
            d3 = tf.layers.dropout(inputs=fc1, rate=dropout_rate, training=self.training)
            fc4 = fully_connected(inputs=d3, num_outputs=self.params.get_int('fc4', 10, 40),
                weights_initializer=he_init, normalizer_fn=tf.layers.batch_normalization, normalizer_params=bn_params)
            d4 = tf.layers.dropout(inputs=fc1, rate=dropout_rate, training=self.training)
            fc5 = fully_connected(inputs=d4, num_outputs=self.params.get_int('fc5', 10, 30),
                weights_initializer=he_init, normalizer_fn=tf.layers.batch_normalization, normalizer_params=bn_params)

            self.logits = fully_connected(inputs=fc5,
                                             num_outputs=self.categories,
                                             activation_fn=None)
            self.predictions = tf.nn.top_k(self.logits, 2)
            self.xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
            self.loss = tf.reduce_mean(self.xentropy)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.grad_update = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.logits, self.y, 1), tf.float32))
            self.saver = tf.train.Saver()

            if cached_model is None:
                self.session.run(tf.global_variables_initializer())
            else:
                self.saver.restore(self.session, cached_model)

    def test_eval(self):
        batch = next(data_gen(self.test_folder, 5000, self.parse_file))
        accuracy = self.run_batch([self.accuracy], batch, False)[0]
        batch = next(data_gen(self.test_folder, 100, self.parse_file))
        training_predictions = self.run_batch([self.predictions], batch, False)[0]
        print('Test Accuracy: {}'.format(accuracy))
        print(batch[0])
        print(training_predictions[1][:,0])
        return accuracy

    def run_batch(self, eval_list, batch, training):
        moves, features = batch
        feed_dict = {self.x: features, self.y:moves, self.training: training}
        return self.session.run(eval_list, feed_dict=feed_dict)

    def train_on_files(self, ckpt_file, epochs):
        map_size = 0
        i = 0
        for x in range(epochs):
            for batch in data_gen(self.train_folder, self.batch_size, self.parse_file):
                t = time.time()
                training_predictions, loss, _, accuracy = self.run_batch(
                    [self.predictions, self.loss, self.grad_update, self.accuracy], batch, True)
                i += 1
                if not (i % 1000):
                    print("batch: {}".format(i))
            accuracy = float(self.test_eval())
            self.params.to_json("params/{}".format(self.params.name), {'accuacy': accuracy})
            self.saver.save(self.session, ckpt_file.format(self.params.name, i))


class MovementModel(HaliteModel):

    def __init__(self, cached_model=None, params_file=None, train_folder=None, test_folder=None):
        super(MovementModel, self).__init__(OUTPUT_SIZE, cached_model, params_file, train_folder, test_folder)

    def parse_file(self, handle):
        _, move, features = pickle.load(handle)
        return MOVE_TO_OUTPUT[move], features

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

class SpawnModel(HaliteModel):

    def __init__(self, cached_model=None, params_file=None, train_folder=None, test_folder=None):
        super(SpawnModel, self).__init__(2, cached_model, params_file, train_folder, test_folder)

    def parse_file(self, handle):
        move, features = pickle.load(handle)
        return int(move), features

    def predict(self, game_state):
        feature_list = []
        feature_list.append(game_state.center_shift())
        feature_map = np.stack(feature_list, axis=0)

        with Timer("Generate Prediction"):
            feed_dict = {self.x: feature_map, self.training: False}
            predictions = self.session.run([self.predictions], feed_dict=feed_dict)[0]
            log_message(predictions)
        _, moves = predictions
        moves = np.ndarray.flatten(moves)
        return bool(moves[0])
