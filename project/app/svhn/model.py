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
"""Builds the CNN network
"""

import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

# Global constants describing the SVHN data set.
tf.app.flags.DEFINE_integer('IMAGE_SIZE', 32, "")
tf.app.flags.DEFINE_integer('NUM_CHANNELS', 1, "")
tf.app.flags.DEFINE_integer('NUM_LABELS', 11, "")
tf.app.flags.DEFINE_integer('N', 5, "")


# Model variables
tf.app.flags.DEFINE_integer('BATCH_SIZE', 64, "")
tf.app.flags.DEFINE_integer('PATCH_SIZE', 5, "")
tf.app.flags.DEFINE_integer('DEPTH_1', 16, "")
tf.app.flags.DEFINE_integer('DEPTH_2', 32, "")
tf.app.flags.DEFINE_integer('DEPTH_3', 64, "")
tf.app.flags.DEFINE_integer('NODES', 256, "")

"""
  Convolutional Neural Netowrk class
  Methods:
  # logits = inference(data)
  # loss = loss(logits, labels)
  # optimizer = train(loss)
  # predictions = predict(data)
"""
class CNN:

  def __init__(self):
    # Variables.
    # 5x5 Filter, depth 16
    self.conv1_weights = tf.Variable(tf.truncated_normal([FLAGS.PATCH_SIZE, FLAGS.PATCH_SIZE, FLAGS.NUM_CHANNELS, FLAGS.DEPTH_1], stddev=0.1))
    self.conv1_biases = tf.Variable(tf.zeros([FLAGS.DEPTH_1]))

    # 5x5 Filter, depth 32
    self.conv2_weights = tf.Variable(tf.truncated_normal([FLAGS.PATCH_SIZE, FLAGS.PATCH_SIZE, FLAGS.DEPTH_1, FLAGS.DEPTH_2], stddev=0.1))
    self.conv2_biases = tf.Variable(tf.constant(1.0, shape=[FLAGS.DEPTH_2]))

    # 5x5 Filter, depth 64
    self.conv3_weights = tf.Variable(tf.truncated_normal([FLAGS.PATCH_SIZE, FLAGS.PATCH_SIZE, FLAGS.DEPTH_2, FLAGS.DEPTH_3], stddev=0.1))
    self.conv3_biases = tf.Variable(tf.constant(1.0, shape=[FLAGS.DEPTH_3]))

    # Linear
    self.N1_weights = tf.Variable(tf.truncated_normal([FLAGS.NODES, FLAGS.NUM_LABELS],stddev=0.1))
    self.N1_biases = tf.Variable(tf.constant(1.0, shape=[FLAGS.NUM_LABELS]))

    self.N2_weights = tf.Variable(tf.truncated_normal([FLAGS.NODES, FLAGS.NUM_LABELS], stddev=0.1))
    self.N2_biases = tf.Variable(tf.constant(1.0, shape=[FLAGS.NUM_LABELS]))

    self.N3_weights = tf.Variable(tf.truncated_normal([FLAGS.NODES, FLAGS.NUM_LABELS], stddev=0.1))
    self.N3_biases = tf.Variable(tf.constant(1.0, shape=[FLAGS.NUM_LABELS]))

    self.N4_weights = tf.Variable(tf.truncated_normal([FLAGS.NODES, FLAGS.NUM_LABELS], stddev=0.1))
    self.N4_biases = tf.Variable(tf.constant(1.0, shape=[FLAGS.NUM_LABELS]))

    self.N5_weights = tf.Variable(tf.truncated_normal([FLAGS.NODES, FLAGS.NUM_LABELS], stddev=0.1))
    self.N5_biases = tf.Variable(tf.constant(1.0, shape=[FLAGS.NUM_LABELS]))

    self.NL_weights = tf.Variable(tf.truncated_normal([FLAGS.NODES, FLAGS.NUM_LABELS], stddev=0.1))
    self.NL_biases = tf.Variable(tf.constant(1.0, shape=[FLAGS.NUM_LABELS]))

  def inference(self, data):
    """Build the SVHN CNN model.

    Args:
      data: dataset of processed SVHN images.

    Returns:
      Logits.
    """
    kernel1 = tf.nn.conv2d(data, self.conv1_weights, strides=[1, 1, 1, 1], padding='VALID')
    conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, self.conv1_biases))
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    norm1 = tf.nn.local_response_normalization(pool1)

    kernel2 = tf.nn.conv2d(norm1, self.conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, self.conv2_biases))
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    norm2 = tf.nn.local_response_normalization(pool2)

    kernel3 = tf.nn.conv2d(norm2, self.conv3_weights, [1, 1, 1, 1], padding='VALID')
    conv3 = tf.nn.relu(tf.nn.bias_add(kernel3, self.conv3_biases))
    norm3 = tf.nn.local_response_normalization(conv3)
    pool = tf.nn.max_pool(norm3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #pool = norm3

    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

    # Linear Extraction for each component
    logits_L = tf.nn.bias_add(tf.matmul(reshape, self.NL_weights), self.NL_biases)
    logits_1 = tf.nn.bias_add(tf.matmul(reshape, self.N1_weights), self.N1_biases)
    logits_2 = tf.nn.bias_add(tf.matmul(reshape, self.N2_weights), self.N2_biases)
    logits_3 = tf.nn.bias_add(tf.matmul(reshape, self.N3_weights), self.N3_biases)
    logits_4 = tf.nn.bias_add(tf.matmul(reshape, self.N4_weights), self.N4_biases)
    logits_5 = tf.nn.bias_add(tf.matmul(reshape, self.N5_weights), self.N5_biases)

    return logits_L, logits_1, logits_2, logits_3, logits_4, logits_5

  def loss(self, logits, labels):
    """Estimate loss applying Softmax and Cross Entropy to each
    output from the CNN

    Args:
      logits: Logits from inference().
      labels: Labels

    Returns:
      Loss tensor
    """
    loss_L = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[0], labels[:,0]))
    loss_1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[1], labels[:,1]))
    loss_2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[2], labels[:,2]))
    loss_3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[3], labels[:,3]))
    loss_4 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[4], labels[:,4]))
    loss_5 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[5], labels[:,5]))

    loss = loss_L + loss_1 + loss_2 + loss_3 + loss_4 + loss_5
    return loss

  def train(self, loss):
    """Train SVHN CNN model.
    Create an optimizer and apply to all trainable variables.

    Args:
      loss: Loss from loss().

    Returns:
      optimizer: op for training.
    """
    optimizer = tf.train.AdagradOptimizer(0.01).minimize(loss)
    return optimizer


  def predict(self, data):
    """Make predictions from data

    Args:
      data: processed data from images

    Returns:
      predictions: probabilities (softmax)
    """
    return tf.pack([
                  tf.nn.softmax(self.inference(data)[0]),\
                  tf.nn.softmax(self.inference(data)[1]),\
                  tf.nn.softmax(self.inference(data)[2]),\
                  tf.nn.softmax(self.inference(data)[3]),\
                  tf.nn.softmax(self.inference(data)[4]),\
                  tf.nn.softmax(self.inference(data)[5])])
