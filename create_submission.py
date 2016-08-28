import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

import models
import re
import glob
import numpy as np
from scipy import misc
from skimage.transform import resize
from itertools import chain
import xgboost as xgb
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.optimizers import SGD
import utils

num = re.compile('\d+')

X_mean = 99.537
X_std = 52.875


#get data
files = glob.glob("./test/*[0-9].tif")

X = np.zeros((len(files), 96, 128))
ids = np.zeros(len(files))
for i in range(len(files)):
    f = files[i]

    X[i, :, :] =  (misc.imresize(misc.imread(f), (96,128)) - X_mean) / X_std

    im_num = num.findall(f)[0]
    ids[i] = im_num

X = np.expand_dims(X, 1)


#load classifiers
model = models.FCN_class('class_full_weights.h5')
full_preds = model.predict(X)

model = models.simple_classifier('simple_full_weights.h5')
simple_preds = model.predict(X)

#load predictions of first flattened layer in VGG
vgg_out = np.load('/mnt/weights/vgg_test_preds.npy')

bst = xgb.Booster({'nthread':4})
bst.load_model('xgb.model')
xgb_preds = bst.predict(xgb.DMatrix(vgg_out))


#load ensemble classifier
meta_in = np.transpose(np.vstack((simple_preds[:,0], full_preds[:,0], xgb_preds)))

bst = xgb.Booster({'nthread':4})
bst.load_model('meta.model')
meta_out = bst.predict(xgb.DMatrix(meta_in))

#cutoff: .6 for net trained on all positive examples, .4 for net trained on full distribution of values 
pred_labels = np.zeros(meta_out.shape[0])
for i in range(len(pred_labels)):
    if meta_out[i] > .6:
        pred_labels[i] = 1


#load FCN model
model = models.FCN_model('unet_positive_full_weights.h5')
X_preds = model.predict(X)

for i in range(len(pred_labels)):
    if pred_labels[i] == 0:
        X_preds[i, 0, :, :] = np.zeros((96, 128))

#predict
X = np.zeros((X_preds.shape[0], 1, 420, 580))

for i in range(X.shape[0]):
    X[i,0,:,:] = resize(X_preds[i,0,:,:], (420,580))

#make submission
def run_length_enc(label):
    x = label.transpose().flatten()
    y = np.where(x > 0.99)[0]

    if len(y) < 300:  # consider as empty
        return ''

    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])

    length = end - start

    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))

    return ' '.join([str(r) for r in res])

with open('submission.csv', 'w+') as f:
    f.write('img,pixels\n')
    for i in range(len(ids)):
        s = str(int(ids[i])) + ',' + run_length_enc(X[i,0,:,:])
        f.write(s + '\n')



