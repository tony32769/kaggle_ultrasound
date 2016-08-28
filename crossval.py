import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.cross_validation import StratifiedKFold
import models
import utils
import xgboost as xgb
from keras.preprocessing.image import ImageDataGenerator
import glob
from scipy import misc
from skimage import transform

#get data
X, y = utils.get_pics("./train/*[0-9]_*[0-9].tif", True)

#define CV
labels = np.zeros(y.shape[0])
for i in range(len(labels)):
    if np.sum(y[i, 0, :, :]) > 0:
        labels[i] = 1

labels = np.expand_dims(labels, 1)

#image preprocessing functions
def trans(im1, im2):
    rands = np.random.rand(4) - .5

    tform = transform.AffineTransform(shear= rands[0] / 4., rotation= rands[1] / 5.,
                                        translation = (rands[2]*20., rands[3]*15.))
    im1 = transform.warp(im1, tform)
    im2 = transform.warp(im2, tform)

    return im1, im2

def generator(X, y):
    
    while 1:
        idx = np.random.choice(range(X.shape[0]), 32)
        X_batch = X[idx, :, :, :]
        y_batch = y[idx, :, :, :]

        for i in range(32):
            X_batch[i, 0, :, :], y_batch[i, 0, :, :] = trans(X_batch[i, 0, :, :], y_batch[i, 0, :, :])
        
        yield X_batch, y_batch


#train FCN with CV
# classifiers trained similarly in ipynb
n_folds = 2

skf = StratifiedKFold(labels[:,0], n_folds=n_folds, shuffle=True)

oos_class_preds = np.zeros(y.shape)

#train with Cross validation
for i, (train, test) in enumerate(skf):
    print "Running Fold", i+1, "/", n_folds

    model = None # Clearing the NN.
    model = models.FCN_model()
    
    train_gen = generator(X[train, :, :, :], y[train, :, :, :])
    
    Checkpoint = ModelCheckpoint('unet_fold_' + str(i) + '_weights.h5', save_best_only=True)
    model.fit_generator(train_gen, 2800, 30,
                      callbacks = [Checkpoint],
                      validation_data = (X[test, :, :, :], y[test, :, :, :]))

    model = models.FCN_model('unet_fold_' + str(i) + '_weights.h5')
    oos_class_preds[test, :,:,:] = model.predict(X[test, :, :, :])


np.save('oos_preds.npy', oos_class_preds)


