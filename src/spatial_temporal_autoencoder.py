import tensorflow as tf
from conv_lstm_cell import ConvLSTMCell

# network architecture definition
NCHANNELS = 1
CONV1 = 128
CONV2 = 64
DECONV1 = 128
DECONV2 = 1
WIDTH = 227
HEIGHT = 227
TVOL = 10
NUM_RNN_LAYERS = 3


class SpatialTemporalAutoencoder(object):
    def __init__(self, alpha, batch_size):
        self.x_ = tf.placeholder(tf.float32, [None, TVOL, HEIGHT, WIDTH, NCHANNELS])
        self.y_ = tf.placeholder(tf.float32, [None, TVOL, HEIGHT, WIDTH, NCHANNELS])
        # usually y_ = x_ if reconstruction error objective

        self.batch_size = batch_size
        W_init = tf.contrib.layers.xavier_initializer_conv2d()
        self.params = {
            "c_w1": tf.get_variable(shape=[11, 11, NCHANNELS, CONV1], initializer=W_init),
            "c_b1": tf.Variable(tf.constant(0.05, dtype=tf.float32, shape=[CONV1])),
            "c_w2": tf.get_variable(shape=[5, 5, CONV1, CONV2], initializer=W_init),
            "c_b2": tf.Variable(tf.constant(0.05, dtype=tf.float32, shape=[CONV2])),
            "c_w3": tf.get_variable(shape=[5, 5, CONV2, DECONV1], initializer=W_init),
            "c_b3": tf.Variable(tf.constant(0.05, dtype=tf.float32, shape=[DECONV1])),
            "c_w4": tf.get_variable(shape=[11, 11, DECONV1, DECONV2], initializer=W_init),
            "c_b4": tf.Variable(tf.constant(0.05, dtype=tf.float32, shape=[DECONV2]))
        }

        self.conved = self.spatial_encoder(self.x_)
        self.convLSTMed = self.temporal_encoder_decoder(self.conved)
        self.y = self.spatial_decoder(self.convLSTMed)
        self.y = tf.reshape(self.y, shape=[-1, TVOL, HEIGHT, WIDTH, NCHANNELS])

        self.reconstruction_loss = tf.reduce_mean(tf.pow(self.y_ - self.y, 2))
        self.regularization_loss = tf.constant(0)
        self.loss = self.reconstruction_loss + self.regularization_loss
        self.optimizer = tf.train.AdamOptimizer(alpha).minimize(self.loss)

        self.saver = tf.train.Saver()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    @staticmethod
    def conv2d(x, w, b, activation=tf.nn.tanh, strides=1):
        """
        Build a convolutional layer
        :param x: input
        :param w: filter
        :param b: bias
        :param activation: activation func
        :param strides: the stride when filter is scanning through image
        :return: a convolutional layer representation
        """
        x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='VALID')
        x = tf.nn.bias_add(x, b)
        return activation(x)

    @staticmethod
    def deconv2d(x, w, b, out_shape, activation=tf.nn.tanh, strides=1):
        """
        Build a deconvolutional layer
        :param x: input
        :param w: filter
        :param b: bias
        :param out_shape: shape of output tensor
        :param activation: activation func
        :param strides: the stride when filter is scanning
        :return: a deconvolutional layer representation
        """
        x = tf.nn.conv2d_transpose(x, w, output_shape=out_shape, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return activation(x)

    def spatial_encoder(self, x):
        """
        Build a spatial encoder that performs convolutions
        :param x: tensor of input image of shape (batch_size, TVOL, HEIGHT, WIDTH, NCHANNELS)
        :return: convolved representation of shape (batch_size * TVOL, h, w, c)
        """
        h, w, c = x.shape[2:]
        x = tf.reshape(x, shape=[-1, h, w, c])
        conv1 = self.conv2d(x, self.params['c_w1'], self.params['c_b1'], activation=tf.nn.tanh, strides=4)
        conv2 = self.conv2d(conv1, self.params['c_w2'], self.params['c_b2'], activation=tf.nn.tanh, strides=2)
        return conv2

    def spatial_decoder(self, x):
        """
        Build a spatial decoder that performs deconvolutions on the input
        :param x: tensor of some transformed representation of input of shape (batch_size, TVOL, h, w, c)
        :return: deconvolved representation of shape (batch_size * TVOL, HEIGHT, WEIGHT, NCHANNELS)
        """
        h, w, c = x.shape[2:]
        x = tf.reshape(x, shape=[-1, h, w, c])
        deconv1 = self.deconv2d(x, self.params['c_w3'], self.params['c_b3'],
                                [self.batch_size * TVOL, self.params['c_w3'].shape[3], 55, 55],
                                activation=tf.nn.tanh, strides=2)
        deconv2 = self.deconv2d(deconv1, self.params['c_w4'], self.params['c_b4'],
                                [self.batch_size * TVOL, self.params['c_w4'].shape[3], HEIGHT, WIDTH],
                                activation=tf.nn.tanh, strides=4)
        return deconv2

    def temporal_encoder_decoder(self, x):
        """
        Build a temporal encoder-decoder network that uses convLSTM layers to perform sequential operation
        :param x: convolved representation of input volume of shape (batch_size * TVOL, h, w, c)
        :return: convLSTMed representation (batch_size, TVOL, h, w, c)
        """
        h, w, c = x.shape[1:]
        x = tf.reshape(x, shape=[-1, TVOL, h, w, c])
        x = tf.unpack(x, axis=1)
        num_filters = [64, 32, 64]
        filter_sizes = [[3, 3] * 3]
        cell = tf.nn.rnn_cell.MultiRNNCell(
            [ConvLSTMCell(shape=[h, w], num_filters=num_filters[i], filter_size=filter_sizes[i])
             for i in xrange(NUM_RNN_LAYERS)])
        states_series, _ = tf.nn.rnn.static_rnn(cell, x, dtype=tf.float32)
        output = tf.transpose(tf.stack(states_series, axis=0), [1, 0, 2, 3, 4])
        return output

    def get_loss(self, x):
        return self.loss.eval(feed_dict={self.x_: x, self.y_: x}, session=self.sess)

    def step(self, x):
        self.sess.run(self.optimizer, feed_dict={self.x_: x, self.y_: x})
