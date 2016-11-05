from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
import numpy as np
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

FLAGS = None


import Config
import random
import pandas

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot
def extract_samples():
  rats1 = pandas.read_csv(FLAGS.data_dir + "postive_sample.csv", header=None)
  len1 = len(rats1)
  rats2 = pandas.read_csv(FLAGS.data_dir + "negtive_sample.csv", header=None)
  len2 = len(rats2)
  items = [i for i in range(len1 + len2)]
  random.shuffle(items)
  feature = []
  label = []
  for item in items:
    if item < len1:
      feature.append([fea for fea in rats1.iloc[item]])
      label.append(0)
    else:
      feature.append([fea for fea in rats2.iloc[item - len1]])
      label.append(1)
  features=np.array(feature)
  labels=dense_to_one_hot(np.array(label),FLAGS.class_numbers)
  #features = features.reshape(num_images, rows, cols, 1)
  return features,labels

class DataSet(object):

  def __init__(self,
               samples,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=False):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert samples.shape[0] == labels.shape[0], (
          'samples.shape: %s labels.shape: %s' % (samples.shape, labels.shape))
      self._num_examples = samples.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        '''
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
        '''
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = samples.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = samples
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

def read_data_sets(train_dir,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   ):
  if fake_data:

    def fake():
      return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)

    train = fake()
    validation = fake()
    test = fake()
    return base.Datasets(train=train, validation=validation, test=test)

  feature,labels=extract_samples()
  i=int(len(feature)*FLAGS.train_ratio)

  train_features_buff= feature[:i]
  train_labels_buff = labels[:i]
  test_samples = feature[i:]
  test_labels = labels[i:]

  if not 0 <= FLAGS.validation_size <= len(train_features_buff):
    raise ValueError(
        'Validation size should be between 0 and {}. Received: {}.'
        .format(len(train_features_buff), FLAGS.validation_size))

  validation_samples = train_features_buff[:FLAGS.validation_size]
  validation_labels = train_labels_buff[:FLAGS.validation_size]
  train_samples = train_features_buff[FLAGS.validation_size:]
  train_labels = train_labels_buff[FLAGS.validation_size:]

  train = DataSet(train_samples, train_labels, dtype=dtype, reshape=reshape)
  validation = DataSet(validation_samples,
                       validation_labels,
                       dtype=dtype,
                       reshape=reshape)
  test = DataSet(test_samples, test_labels, dtype=dtype, reshape=reshape)

  return base.Datasets(train=train, validation=validation, test=test)
def train():
  # Import data
  mnist = read_data_sets(FLAGS.data_dir,
                                    one_hot=True,
                                    fake_data=FLAGS.fake_data)

  sess = tf.InteractiveSession()
  # Create a multilayer model.

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, FLAGS.feature_numbers], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, FLAGS.class_numbers], name='y-input')

  #with tf.name_scope('input_reshape'):
  #  image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
  #  tf.summary.image('input', image_shaped_input, 10)

  # We can't initialize these variables to 0 - the network will get stuck.
  def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)
      return activations

  hidden1 = nn_layer(x, FLAGS.feature_numbers, 40, 'layer1')

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

  # Do not apply softmax activation yet, see below.
  y = nn_layer(dropped, 40, FLAGS.class_numbers, 'layer2', act=tf.identity)

  with tf.name_scope('cross_entropy'):
    # The raw formulation of cross-entropy,
    #
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                               reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the
    # raw outputs of the nn_layer above, and then average across
    # the batch.
    diff = tf.nn.softmax_cross_entropy_with_logits(y, y_)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
  tf.summary.scalar('cross_entropy', cross_entropy)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
  merged = tf.summary.merge_all()
  train_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/train',
                                        sess.graph)
  test_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/test')
  tf.initialize_all_variables().run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(FLAGS.batch_sizes, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  for i in range(FLAGS.max_steps):
    if i % 10 == 0:  # Record summaries and test-set accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:  # Record train set summaries, and train
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)

  train_writer.close()
  test_writer.close()


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument('--feature_numbers', type=int, default=17,
                      help='the feature numbers of input data')
  parser.add_argument('--class_numbers', type=int, default=2,
                      help='the class numbers ')
  parser.add_argument('--batch_sizes', type=int, default=10000,
                      help='the class numbers ')
  parser.add_argument('--train_ratio', type=float, default=0.9,
                      help='ratio of samples used for train samples,remains used for test samples')
  parser.add_argument('--validation_size', type=int, default=10000,
                      help='')
  parser.add_argument('--data_dir', type=str, default=Config.sample_path,
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default='/tmp/AGame/summaries',
                      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run(main=main)
