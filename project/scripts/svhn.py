# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the SVHN network

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import gzip
import os
import re
import sys
import tarfile
import h5py
import glob

import numpy as np
import tensorflow as tf
from scipy import ndimage
from scipy.misc import imresize
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from PIL import Image

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '../data/SVHN_data',
                           """Path to the SVHN data directory.""")

# Global constants describing the SVHN data set.
IMAGE_SIZE = 32 # 32x32 pixel each image
NUM_CHANNELS = 1 # grayscale
NUM_CLASSES = 11 # 0-9 (10) and 10 that means 'no digit'
N = 5 # max number of digits in a sequence

# Data consts
DATA_URL = 'http://ufldl.stanford.edu/housenumbers/'
DATA_FOLDER = '../data/SVHN_data/'
TRAIN_FILENAME= 'train.tar.gz'
TEST_FILENAME = 'test.tar.gz'
EXTRA_FILENAME = 'extra.tar.gz'

def inference(data):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # Variables.
  # 5x5 Filter, depth 16
  conv1_weights = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS, DEPTH_1], stddev=0.1))
  conv1_biases = tf.Variable(tf.zeros([DEPTH_1]))

  # 5x5 Filter, depth 32
  conv2_weights = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, DEPTH_1, DEPTH_2], stddev=0.1))
  conv2_biases = tf.Variable(tf.constant(1.0, shape=[DEPTH_2]))

  # 5x5 Filter, depth 64
  conv3_weights = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, DEPTH_2, DEPTH_3], stddev=0.1))
  conv3_biases = tf.Variable(tf.constant(1.0, shape=[DEPTH_3]))

  # Linear
  N1_weights = tf.Variable(tf.truncated_normal([HIDDEN_NODES, NUM_LABELS],stddev=0.1))
  N1_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]))

  N2_weights = tf.Variable(tf.truncated_normal([HIDDEN_NODES, NUM_LABELS], stddev=0.1))
  N2_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]))

  N3_weights = tf.Variable(tf.truncated_normal([HIDDEN_NODES, NUM_LABELS], stddev=0.1))
  N3_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]))

  N4_weights = tf.Variable(tf.truncated_normal([HIDDEN_NODES, NUM_LABELS], stddev=0.1))
  N4_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]))

  N5_weights = tf.Variable(tf.truncated_normal([HIDDEN_NODES, NUM_LABELS], stddev=0.1))
  N5_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]))

  NL_weights = tf.Variable(tf.truncated_normal([HIDDEN_NODES, NUM_LABELS], stddev=0.1))
  NL_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]))

  # Model.
  def model(data):

    kernel1 = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='VALID')
    conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, conv1_biases))
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    norm1 = tf.nn.local_response_normalization(pool1)

    kernel2 = tf.nn.conv2d(norm1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, conv2_biases))
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    norm2 = tf.nn.local_response_normalization(pool2)

    kernel3 = tf.nn.conv2d(norm2, conv3_weights, [1, 1, 1, 1], padding='VALID')
    conv3 = tf.nn.relu(tf.nn.bias_add(kernel3, conv3_biases))
    norm3 = tf.nn.local_response_normalization(conv3)
    pool = tf.nn.max_pool(norm3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #pool = norm3

    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

    # Linear Extraction for each component
    logits_L = tf.nn.bias_add(tf.matmul(reshape, NL_weights), NL_biases)
    logits_1 = tf.nn.bias_add(tf.matmul(reshape, N1_weights), N1_biases)
    logits_2 = tf.nn.bias_add(tf.matmul(reshape, N2_weights), N2_biases)
    logits_3 = tf.nn.bias_add(tf.matmul(reshape, N3_weights), N3_biases)
    logits_4 = tf.nn.bias_add(tf.matmul(reshape, N4_weights), N4_biases)
    logits_5 = tf.nn.bias_add(tf.matmul(reshape, N5_weights), N5_biases)

    return logits_L, logits_1, logits_2, logits_3, logits_4, logits_5

   logits = model(data)
   return logits


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  loss_L = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[0], labels[:,0]))
  loss_1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[1], labels[:,1]))
  loss_2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[2], labels[:,2]))
  loss_3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[3], labels[:,3]))
  loss_4 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[4], labels[:,4]))
  loss_5 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[5], labels[:,5]))

  loss = loss_L + loss_1 + loss_2 + loss_3 + loss_4 + loss_5
  return loss


def train(total_loss):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  optimizer = tf.train.AdagradOptimizer(0.01).minimize(total_loss)


def load_data():
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  pickle_file = FLAGS.data_dir + '/' + 'SVHN-seq.pickle'
  if os.path.isfile(pickle_file):
    print('Pickle detected, avoiding download and processing')
    svhn_input.load_data_pickle(pickle_file=pickle_file)
  else:
    get_data()

def get_data():
  pass
  # if not data:
  #   print('Pickle not found... Download, Extracting and Preprocessing Info')
  #   print('This maybe take a while...')
  #   train_filename = svhn_input.maybe_download(TRAIN_FILENAME, 404141560)
  #   test_filename = svhn_input.maybe_download(TEST_FILENAME, 276555967)
  #   extra_filename = svhn_input.maybe_download(EXTRA_FILENAME, 1955489752)
  #
  #   train_folder = svhn_input.maybe_extract(train_filename)
  #   test_folder = svhn_input.maybe_extract(test_filename)
  #   extra_folder = svhn_input.maybe_extract(extra_filename)
  #
  #   train_folder = train_folder+'/'
  #   test_folder = test_folder+'/'
  #   extra_folder = extra_folder+'/'
  #processing... etc
