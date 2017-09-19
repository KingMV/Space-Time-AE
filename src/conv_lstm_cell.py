import tensorflow as tf


class ConvLSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, shape, num_filters, filter_size):
        """
        :param shape: (list) spatial dimensions [H, W]
        :param num_filters: (int) number of output feature maps
        :param filter_size: (list) dims of filter [F, F]
        """
        self.shape = shape
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.size = tf.TensorShape(shape + [self.num_filters])
        self.feature_axis = self.size.ndims

    def call(self, x, state):
        """
        Perform convLSTM cell ops given input at a given state
        :param x: (tensor) input image of shape [batch_size, timesteps, H, W, channels]
        :param state: (tuple) previous memory and hidden states of the cell
        :return: new state after performing convLSTM ops given input and previous state
        """
        c, h = state

        x = tf.concat([x, h], axis=self.feature_axis)
        n = x.shape[-1]
        m = 4 * self.num_filters if self.num_filters > 1 else 4
        W = tf.get_variable("filter", self.filter_size + [n, m])
        y = tf.nn.convolution(x, W, padding="SAME")
        y += tf.get_variable("bias", [m], initializer=tf.zeros_initializer())
        j, i, f, o = tf.split(y, 4, axis=self.feature_axis)

        f = tf.sigmoid(f)
        i = tf.sigmoid(i)
        c = c * f + i * tf.nn.tanh(j)

        o = tf.sigmoid(o)
        h = o * tf.nn.tanh(c)

        state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

        return h, state
