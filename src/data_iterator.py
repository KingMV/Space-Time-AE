import numpy as np

FRAMES_PER_VIDEO = 200
TVOL = 10


class DataIterator(object):
    def __init__(self, p_train, p_test, p_labels):
        self.train, self.test, self.labels = np.load(p_train), np.load(p_test), np.load(p_labels)

    def get_train_batch(self, batch_size):
        batch = np.zeros(shape=(batch_size, TVOL) + self.train[0].shape)
        for i in xrange(batch_size):
            vid_idx = np.random.randint(0, self.train.shape[0] / FRAMES_PER_VIDEO)
            aug_idx = np.random.randint(1, 4)
            frame_idx = np.random.randint(0, FRAMES_PER_VIDEO - TVOL * aug_idx)
            batch[i] = self.train[(FRAMES_PER_VIDEO * vid_idx + frame_idx):
                                  (FRAMES_PER_VIDEO * vid_idx + frame_idx + TVOL * aug_idx):aug_idx]
        return batch

    def get_test_batch(self, batch_size):
        batch = np.zeros(shape=(batch_size, TVOL) + self.test[0].shape)
        for i in xrange(batch_size):
            vid_idx = np.random.randint(0, self.train.shape[0] / FRAMES_PER_VIDEO)
            aug_idx = np.random.randint(1, 4)
            frame_idx = np.random.randint(0, FRAMES_PER_VIDEO - TVOL * aug_idx)
            batch[i] = self.train[(FRAMES_PER_VIDEO * vid_idx + frame_idx):
                                  (FRAMES_PER_VIDEO * vid_idx + frame_idx + TVOL * aug_idx):aug_idx]
        return self.test, self.labels

    @property
    def get_train_size(self):
        return self.train.shape[0]

    @property
    def get_test_size(self):
        return self.test.shape[0]
