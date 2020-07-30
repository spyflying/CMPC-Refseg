'''
vgg16 model with atrous & fully convolution layers
'''

import tensorflow as tf

class Vgg16:
  def __init__(self, data):
    self.data = data
    '''
    build graph
    '''
    self.conv1_1 = self.conv_relu('conv1_1', self.data,    3,  64)
    self.conv1_2 = self.conv_relu('conv1_2', self.conv1_1, 64, 64)
    self.pool1 = self.max_pool('pool1', self.conv1_2)

    self.conv2_1 = self.conv_relu('conv2_1', self.pool1,   64,  128)
    self.conv2_2 = self.conv_relu('conv2_2', self.conv2_1, 128, 128)
    self.pool2 = self.max_pool('pool2', self.conv2_2)

    self.conv3_1 = self.conv_relu('conv3_1', self.pool2,   128, 256)
    self.conv3_2 = self.conv_relu('conv3_2', self.conv3_1, 256, 256)
    self.conv3_3 = self.conv_relu('conv3_3', self.conv3_2, 256, 256)
    self.pool3 = self.max_pool('pool3', self.conv3_3)

    self.conv4_1 = self.conv_relu('conv4_1', self.pool3,   256, 512)
    self.conv4_2 = self.conv_relu('conv4_2', self.conv4_1, 512, 512)
    self.conv4_3 = self.conv_relu('conv4_3', self.conv4_2, 512, 512)

    self.conv5_1 = self.conv_relu('conv5_1', self.conv4_3, 512, 512)
    self.conv5_2 = self.conv_relu('conv5_2', self.conv5_1, 512, 512)
    self.conv5_3 = self.conv_relu('conv5_3', self.conv5_2, 512, 512)

    self.fc6 = self.conv_relu('fc6', self.conv5_3, 512, 4096, kernel_size=7)
    self.fc7 = self.conv_relu('fc7', self.fc6, 4096, 4096, kernel_size=1)
    self.fc8 = self.conv_layer('fc8', self.fc7, 4096, 1000, kernel_size=1)

  def max_pool(self, name, bottom, kernel_size=2, stride=2):
    pool = tf.nn.max_pool(bottom, ksize=[1, kernel_size, kernel_size, 1],
              strides=[1, stride, stride, 1], padding='SAME', name=name)
    return pool

  def conv_layer(self, name, bottom, input_dim, output_dim, kernel_size=3, stride=1):
    with tf.variable_scope(name):
      w = tf.get_variable('weights', [kernel_size, kernel_size, input_dim, output_dim],
              initializer=tf.contrib.layers.xavier_initializer_conv2d())
      b = tf.get_variable('biases', output_dim, initializer=tf.constant_initializer(0.))

    conv = tf.nn.conv2d(bottom, w, [1, stride, stride, 1], padding='SAME')
    conv = tf.nn.bias_add(conv, b)
    return conv

  def conv_relu(self, name, bottom, input_dim, output_dim, kernel_size=3, stride=1):
    conv = self.conv_layer(name, bottom, input_dim, output_dim, kernel_size, stride)
    return tf.nn.relu(conv)

  def atrous_conv_relu(self, name, bottom, input_dim, output_dim, kernel_size=3, rate=1):
    with tf.variable_scope(name):
      w = tf.get_variable('weights', [kernel_size, kernel_size, input_dim, output_dim],
              initializer=tf.random_normal_initializer(stddev=0.01))
      b = tf.get_variable('biases', output_dim, initializer=tf.constant_initializer(0.))

    conv = tf.nn.atrous_conv2d(bottom, w, rate=rate, padding='SAME')
    conv = tf.nn.bias_add(conv, b)
    relu = tf.nn.relu(conv)
    return relu
