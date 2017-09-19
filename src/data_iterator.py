import numpy as np

class DataIterator(object):
    def __init__(self, p_train, p_test, p_labels):
        self.train, self.test, self.labels = np.load(p_train), np.load(p_test), np.load(p_labels)

    def get_train_batch(self, batch_size):
        batch = np.zeros(shape=(batch_size, 10) + self.train[0].shape)
        for i in xrange(batch_size):
            vid_idx = np.random.randint(0, self.train.shape[0] / 200)
            aug_idx = np.random.randint(1, 4)
            frame_idx = np.random.randint(0, 200 - 10 * aug_idx)
            batch[i] = self.train[(200 * vid_idx + frame_idx):(200 * vid_idx + frame_idx + 10 * aug_idx):aug_idx]
        return batch
