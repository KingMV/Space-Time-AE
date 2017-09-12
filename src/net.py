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

class network(object):
	def __init__(self, alpha, keep_prob):
		self.x_ = tf.placeholder(tf.float32, [None, WIDTH * HEIGHT * TVOL])
		self.y_ = tf.placeholder(tf.float32, [None, WIDTH * HEIGHT * TVOL]) # usually y_ = x_ if reconstruction

		# encoder
		self.W_conv1 = tf.Variable(, dtype = tf.float32)
		self.b_conv1 = tf.Variable(, dtype = tf.float32)

		# decoder


		self.y = 

		self.reconstruction_loss = tf.reduce_mean(tf.pow(y_ - y, 2))
		self.loss = self.reconstruction_loss
		self.optimizer = tf.train.AdamOptimizer(alpha).minimize(self.loss)

		self.sess = tf.InteractiveSession()
		self.sess.run(tf.global_variables_initializer())

	def decode(self, X):


	def reconstruct(self, Z):


	def get_loss(self, X, Y):
		return self.loss.eval(feed_dict = {self.x_ = X, self.y_ = Y}, session = self.sess)

