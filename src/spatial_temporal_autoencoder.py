from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np

# network architecture definition
NCHANNELS = 1
conv1 = 128
conv2 = 64
deconv1 = 128
deconv2 = 1
WIDTH = 227
HEIGHT = 227
TVOL = 10

class SpatialTemporalAutoencoder(object):
    def __init__(self, alpha, batch_size, image_shape, time_steps, keep_prob):
        self.x_ = tf.placeholder(tf.float32, [None, TVOL, HEIGHT, WIDTH, NCHANNELS])
        self.y_ = tf.placeholder(tf.float32, [None, TVOL, HEIGHT, WIDTH, NCHANNELS]) # usually y_ = x_ if reconstruction
        self.image_shape = image_shape


        self.batch_size = batch_size
        self.params = {
            # 11x11 conv,  input, 32 output
            'c_w1': tf.Variable(tf.random_normal([11, 11, 1 , 128])),
            'c_b1': tf.Variable(tf.random_normal([128])),
            'c_w2': tf.Variable(tf.random_normal([5, 5, 128, 64])),
            'c_b2': tf.Variable(tf.random_normal([64])),
            'c_w3': tf.Variable(tf.random_normal([5, 5,64, 128])),
            'c_b3': tf.Variable(tf.random_normal([128])),
            'c_w4': tf.Variable(tf.random_normal([11, 11, 128, 1])),
            'c_b4': tf.Variable(tf.random_normal([1]))
        }

        self.y =

        self.reconstruction_loss = tf.reduce_mean(tf.pow(y_ - y, 2))
        self.loss = self.reconstruction_loss
        self.optimizer = tf.train.AdamOptimizer(alpha).minimize(self.loss)

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    @staticmethod
    def conv2d(x, w, b, activation, strides = 1):
        """
        Build a convolutional layer

        :param x: input
        :param w: weight matrix
        :param b: bia
        :param activation: activation func
        :param strides: the stride when filter is scanning through image
        :return: a convolutional layer representation
        """
        x = tf.nn.conv2d(x, w, strides = [1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return activation(x)

    def spatial_encoder(self, x):
        """
        Build a spatial encoder network

        :param x: tf tensor
        :return:
        """
        h, w, c = self.image_shape
        x = tf.reshape(x, shape=[-1, h, w, c])
        conv1 = self.conv2d(x, self.params['c_w1'], self.params['c_b1'], activation=tf.nn.relu, strides=4)
        conv2 = self.conv2d(conv1, self.params['c_w2'], self.params['c_b2'], activation=tf.nn.relu, strides=2)

        return conv2

    def spatial_decoder(self, x):
        """
        Build a spatial decoder network
        :param x:
        :return:
        """
        # revise this
        conv3 = self.conv2d(x, self.params['c_w3'], self.params['c_b3'], activation=tf.nn.relu, strides=2)
        conv4 = self.conv2d(conv3, self.params['c_w4'], self.params['c_b4'], activation=tf.nn.relu, strides=4)

        return conv4



    def get_loss(self, x, y):
        return self.loss.eval(feed_dict = {self.x_=y, self.y_ = y}, session = self.sess)

