from tensorflow.nn import dynamic_rnn, bidirectional_dynamic_rnn

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope

from player.constants import MAX_BOARD_SIZE

import tensorflow as tf

def _shape(tensor):
  return tensor.get_shape().as_list()

def grid_to_sequence(tensor):
  """Convert a batch of grids into a batch of sequences.
  Args:
    tensor: a (batch_size, height, width, depth) tensor
  Returns:
    (width, batch_size*height, depth) sequence tensor
  """

  batch_size = tf.shape(tensor)[0]
  _, height, width, depth = _shape(tensor)
  transposed = array_ops.transpose(tensor, [2, 0, 1, 3])
  reshaped = array_ops.reshape(transposed,
                           [width, batch_size * height, depth])
  return reshaped

def sequence_to_grid(tensor, batch_size, height):
  """Convert a batch of sequences into a batch of grids.
  Args:
    tensor: (width, batch_size*height, depth) sequence tensor
    batch_size: the number of image batches
  Returns:
    (batch_size, height, width, depth) tensor
  """

  width, _, depth = _shape(tensor)
  reshaped = array_ops.reshape(tensor,
                               [width, batch_size, height, depth])
  return array_ops.transpose(reshaped, [1, 2, 0, 3])

def bidirectional_horizontal_lstm(cell, num_units, inputs, seq_lengths, scope=None):
    he_init = tf.contrib.layers.variance_scaling_initializer()
    with variable_scope.variable_scope(scope, "BiHorizontalLstm", [inputs]):
        batch_size = tf.shape(inputs)[0]
        height = _shape(inputs)[1]
        sequence = grid_to_sequence(inputs)

        forward_cell = cell(num_units)
        backward_cell = cell(num_units)
        outputs, states = bidirectional_dynamic_rnn(
            forward_cell, backward_cell, sequence, sequence_length=seq_lengths, time_major=True, dtype=tf.float32)
        stacked_state = tf.expand_dims(array_ops.concat(states, 1), 0)
        output = sequence_to_grid(stacked_state, batch_size, height)
        return output


def separable_lstm(cell, num_units, inputs, seq_lengths1, seq_lengths2, scope=None):
  """Run bidirectional LSTMs first horizontally then vertically.
  Args:
    cell: an RNN cell
    num_units: number of neurons
    inputs: input sequence (length, batch_size, ninput)
    sequence_lengths: array of length 'batch_size' containing sequence_lengths
    scope: optional scope name
  Returns:
    (batch_size, height, width, num_units*2) tensor
  """
  with variable_scope.variable_scope(scope, "SeparableLstm", [inputs]):
    hidden = bidirectional_horizontal_lstm(cell, num_units, inputs, seq_lengths1)
    with variable_scope.variable_scope("vertical"):
      transposed = array_ops.transpose(hidden, [0, 2, 1, 3])
      output_transposed = bidirectional_horizontal_lstm(cell, num_units, transposed, seq_lengths2)
    output = array_ops.transpose(output_transposed, [0, 2, 1, 3])
    return output

def separable_lstm2(cell, num_units, inputs, seq_lengths1, seq_lengths2, scope=None):
  """Run bidirectional LSTMs first horizontally then vertically.
  Args:
    cell: an RNN cell
    num_units: number of neurons
    inputs: input sequence (length, batch_size, ninput)
    sequence_lengths: array of length 'batch_size' containing sequence_lengths
    scope: optional scope name
  Returns:
    (batch_size, height, width, num_units*2) tensor
  """
  with variable_scope.variable_scope(scope, "SeparableLstm", [inputs]):
    batch_size = tf.shape(inputs)[0]
    _, height, width, depth = _shape(inputs)
    reshaped = array_ops.reshape(inputs, [batch_size * width, height, depth])
    _, states = bidirectional_dynamic_rnn(
        cell(num_units), cell(num_units), reshaped, sequence_length=seq_lengths1, dtype=tf.float32)
    stacked_state = array_ops.concat(states, 1)
    with variable_scope.variable_scope("vertical"):
        unpacked = array_ops.reshape(stacked_state, [batch_size, width, num_units*2])
        _, states = bidirectional_dynamic_rnn(
            cell(num_units), cell(num_units), unpacked, sequence_length=seq_lengths2, dtype=tf.float32)
    return array_ops.concat(states, 1)
