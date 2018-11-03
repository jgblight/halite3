from tensorflow.nn import dynamic_rnn, bidirectional_dynamic_rnn

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope

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

def sequence_to_grid(tensor, batch_size):
  """Convert a batch of sequences into a batch of grids.
  Args:
    tensor: (width, batch_size*height, depth) sequence tensor
    batch_size: the number of image batches
  Returns:
    (batch_size, height, width, depth) tensor
  """

  width, _, depth = _shape(tensor)
  height = width  # relying on squareness
  reshaped = array_ops.reshape(tensor,
                               [width, batch_size, height, depth])
  return array_ops.transpose(reshaped, [1, 2, 0, 3])


def ndlstm_base_dynamic(cell, num_units, inputs, seq_lengths, scope=None, reverse=False):
  """Run an LSTM, either forward or backward.
  This is a 1D LSTM implementation using dynamic_rnn and
  the TensorFlow LSTM op.
  Args:
    cell: an RNN cell
    num_units: number of neurons
    inputs: input sequence (length, batch_size, ninput)
    scope: optional scope name
    reverse: run LSTM in reverse
  Returns:
    Output sequence (length, batch_size, noutput)
  """
  with variable_scope.variable_scope(scope, "SeqLstm", [inputs]):
    batch_size = tf.shape(inputs)[1]
    lstm_cell = cell(num_units)
    state = array_ops.zeros([batch_size, lstm_cell.state_size])
    if reverse:
      inputs = array_ops.reverse_v2(inputs, [0])
    outputs, states = dynamic_rnn(
        lstm_cell, inputs, initial_state=state, sequence_length=seq_lengths, time_major=True, dtype=tf.float32)
    if reverse:
      outputs = array_ops.reverse_v2(outputs, [0])
    return outputs, states


def horizontal_lstm(cell, num_units, inputs, seq_lengths, scope=None):
  """Run an LSTM bidirectionally over all the rows of each image.
  Args:
    cell: an RNN cell
    num_units: number of neurons
    inputs: input sequence (length, batch_size, ninput)
    scope: optional scope name
  Returns:
    (batch_size, height, width, num_units*2) tensor, where
    num_steps is width and new num_batches is num_image_batches * height
  """
  with variable_scope.variable_scope(scope, "HorizontalLstm", [inputs]):
    batch_size = tf.shape(inputs)[0]
    sequence = grid_to_sequence(inputs)
    with variable_scope.variable_scope("lr"):
      hidden_sequence_lr, _ = ndlstm_base_dynamic(cell, num_units, sequence, seq_lengths)
    with variable_scope.variable_scope("rl"):
      hidden_sequence_rl, _ = (ndlstm_base_dynamic(cell, num_units,
          sequence, seq_lengths, reverse=1))
    outputs = array_ops.concat([hidden_sequence_lr, hidden_sequence_rl],
                                       2)
    output = sequence_to_grid(hidden_sequence_lr, batch_size)
    return output

def bidirectional_horizontal_lstm(cell, num_units, inputs, seq_lengths, scope=None):
    with variable_scope.variable_scope(scope, "BiHorizontalLstm", [inputs]):
        batch_size = tf.shape(inputs)[0]
        sequence = grid_to_sequence(inputs)

        forward_cell = cell(num_units)
        backward_cell = cell(num_units)
        outputs, states = bidirectional_dynamic_rnn(
            forward_cell, backward_cell, sequence, sequence_length=seq_lengths, time_major=True, dtype=tf.float32)

        outputs = array_ops.concat(outputs, 2)
        output = sequence_to_grid(outputs, batch_size)
        return output


def separable_lstm(cell, num_units, inputs, seq_lengths, scope=None):
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
    hidden = bidirectional_horizontal_lstm(cell, num_units, inputs, seq_lengths)
    with variable_scope.variable_scope("vertical"):
      transposed = array_ops.transpose(hidden, [0, 2, 1, 3])
      output_transposed = bidirectional_horizontal_lstm(cell, num_units, transposed, seq_lengths)
    output = array_ops.transpose(output_transposed, [0, 2, 1, 3])
    return output
