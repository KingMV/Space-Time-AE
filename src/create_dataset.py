from PIL import Image
import os
from glob import glob
import numpy as np
from scipy.misc import imresize as resize

train_dir = "/Users/tnybny/Documents/Anomaly detection in video/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train"
test_dir = "/Users/tnybny/Documents/Anomaly detection in video/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test"

# load images and resize to 227 x 227
train = [resize(np.array(Image.open(y)), (227, 227)) for x in os.walk(train_dir) for y in glob(os.path.join(x[0],
                                                                                                            '*.tif'))]
test = [resize(np.array(Image.open(y)), (227, 227)) for x in os.walk(test_dir) for y in glob(os.path.join(x[0],
                                                                                                          '*.tif'))]

# rescale to [0, 1]
train, test = [x / 255. for x in train], [x / 255. for x in test]
train, test = np.asarray(train), np.asarray(test)

# centering
tr_mu, tr_sigma = np.mean(train, axis=0), np.std(train, axis=0)
train, test = (train - tr_mu) / tr_sigma, (test - tr_mu) / tr_sigma

np.save('../data/train.npy', train), np.save('../data/test.npy', test)