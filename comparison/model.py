from keras.models import Sequential
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Activation
from keras.layers import Input
from keras import optimizers
from keras.layers.core import Dropout
import logging
import numpy as np
from comparison.util import cal_per_frame_error


class STAE(object):
    """
    This class tries to rebuild the model in the paper
    """
    def __init__(self, data_shape, learning_rate, optimizer='sgd', loss='mean_squared_error', epsilon=1e-8):
        self.data_shape = data_shape
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss = loss
        self.epsilon = epsilon
        self.model = self.build_model()


    def build_model(self, dropout=0.5):
        model = Sequential()

        # convolutional layer 1
        #model.add(TimeDistributed(BatchNormalization(), input_shape=self.data_shape))
        model.add(TimeDistributed(Conv2D(128, kernel_size=(11, 11), padding='same', strides=(4,4), name='conv1'),
                                  input_shape=self.data_shape))
        #model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))
        model.add(Dropout(dropout))

        # convolutional layer 2
        model.add(TimeDistributed(Conv2D(64, kernel_size=(5,5), padding='same', strides=(2,2), name='conv2')))
        #model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))
        model.add(Dropout(dropout))

        # ConvLSTM
        model.add(ConvLSTM2D(64, kernel_size=(3,3), padding='same', return_sequences=True, name='convlstm1'))
        model.add(ConvLSTM2D(32, kernel_size=(3,3), padding='same', return_sequences=True, name='convlstm2'))
        model.add(ConvLSTM2D(64, kernel_size=(3,3), padding='same', return_sequences=True, name='convlstm3'))

        # deconvolutional layer 1
        model.add(TimeDistributed(Conv2DTranspose(128, kernel_size=(5,5),
                                                  padding='same', strides=(2,2), name='deconv1')))
        #model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))
        #model.add(Dropout(dropout))

        # deconvolutional layer 2
        model.add(TimeDistributed(Conv2DTranspose(1, kernel_size=(11, 11),
                                                  padding='same', strides=(4,4), name='deconv2')))

        self.compile_model(model)
        return model

    def compile_model(self, model):
        if self.optimizer == 'sgd':
            opt = optimizers.SGD(lr=self.learning_rate, nesterov=True)
        elif self.optimizer == 'adam':
            opt = optimizers.Adam(lr=self.learning_rate, epsilon=self.epsilon)
        else:
            raise NotImplementedError('Optimizer {} has not been implemented'.format(self.optimizer))

        model.compile(loss='mean_squared_error', optimizer=opt)
        logging.info("Building model succeeds!")
        return

    def batch_train(self, x_train, y_train):
        train_loss = self.model.train_on_batch(x_train, y_train)
        return train_loss

    def batch_predict(self, x):
        return self.model.predict_on_batch(x)

    def save_model(self):
        pass

    def get_recon_errors(self, test_batch, is_training=False):
        test_batch_prediction = self.batch_predict(test_batch)
        return cal_per_frame_error(test_batch, test_batch_prediction)