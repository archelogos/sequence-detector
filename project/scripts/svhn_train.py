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
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import gzip
import os
import re
import sys

import tensorflow as tf
import svhn

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'trained_models',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 101,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def train():

  with tf.Graph().as_default():
    # Get data and labels
    SVHN_data = svhn.get_data()
    train_dataset = SVHN_data['train_dataset']
    train_labels = SVHN_data['train_labels']
    valid_dataset = SVHN_data['valid_dataset']
    valid_labels = SVHN_data['valid_labels']
    test_dataset = SVHN_data['test_dataset']
    test_labels = SVHN_data['test_labels']

    # Input data.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(FLAGS.BATCH_SIZE, FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, FLAGS.NUM_CHANNELS))
    tf_train_labels = tf.placeholder(tf.int64, shape=(FLAGS.BATCH_SIZE, FLAGS.N+1))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    CNN = svhn.CNN()
    logits = CNN.inference(tf_train_dataset)

    # Calculate loss.
    loss = CNN.loss(logits, tf_train_labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    optimizer = CNN.train(loss)

    # Preds to estimate
    train_prediction = CNN.predict(tf_train_dataset)
    test_prediction = CNN.predict(tf_test_dataset)
    valid_prediction = CNN.predict(tf_valid_dataset)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    #saver.restore(session, os.path.join(FLAGS.train_dir, 'model.ckpt'))
    #print('Model Restored')
    #print('Initialized')

    for step in range(FLAGS.max_steps):
      offset = (step * FLAGS.BATCH_SIZE) % (train_labels.shape[0] - FLAGS.BATCH_SIZE)
      batch_data = train_dataset[offset:(offset + FLAGS.BATCH_SIZE), :, :, :]
      batch_labels = train_labels[offset:(offset + FLAGS.BATCH_SIZE),:]
      feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
      _, l, predictions = sess.run(
        [optimizer, loss, train_prediction], feed_dict=feed_dict)
      if (step % 100 == 0):
        print('Minibatch loss at step %d: %f' % (step, l))
        #print('Minibatch accuracy: %.1f%%' % svhn.accuracy(predictions, batch_labels))
        #print('Validation accuracy: %.1f%%' % svhn.accuracy(valid_prediction.eval(session=sess), valid_labels))

    print('Out of training')
    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
    sr = saver.save(sess, checkpoint_path)
    print('Model Saved')
    #print('Evaluating the final performance')
    #print('Test accuracy: %.1f%%' % svhn.accuracy(test_prediction.eval(session=sess), test_labels))


def main(argv=None):  # pylint: disable=unused-argument
  svhn.load_data()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir, 0777) # force permissions
  train()

if __name__ == '__main__':
  tf.app.run()
