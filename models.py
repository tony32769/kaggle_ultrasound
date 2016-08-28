from keras.models import Model
from keras.layers import Input, merge
from keras.regularizers import l1l2
from keras.layers.core import Flatten, Dense, Dropout, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, RMSprop, Adamax
from keras.layers.advanced_activations import ELU
import numpy as np
from scipy import misc
import glob
from matplotlib import pyplot as plt
from keras import backend as K

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)



#image segmentation based on FCN-8 from Long and Shelhamer
#differences: use ELU instead of ReLU as activation
#           use batch normalization instead of dropout
#           upsample one more time, giving us 'FCN-4' in effect
#           add convolutions at the end to learn smooth boundaries

def FCN_model(weights_path=None, train_all = True):


    inputs = Input(shape=(1,96,128))

    conv1 = Convolution2D(32, 3, 3, border_mode='same', trainable = train_all)(inputs)
    conv1 = ELU()(conv1)
    conv1 = BatchNormalization(axis=1, mode=0)(conv1)
    #conv1 = Dropout(0.1)(conv1)
    conv1 = Convolution2D(32, 3, 3, border_mode='same', trainable = train_all)(conv1)
    conv1 = ELU()(conv1)
    conv1 = BatchNormalization(axis=1, mode=0)(conv1)

    pool1 = MaxPooling2D((2,2), strides=(2,2))(conv1)

    conv2 = Convolution2D(64, 3, 3, border_mode='same', trainable = train_all)(pool1)
    conv2 = ELU()(conv2)
    conv2 = BatchNormalization(axis=1, mode=0)(conv2)
    #conv2 = Dropout(0.1)(conv2)
    conv2 = Convolution2D(64, 3, 3, border_mode='same', trainable = train_all)(conv2)
    conv2 = ELU()(conv2)
    conv2 = BatchNormalization(axis=1, mode=0)(conv2)

    pool2 = MaxPooling2D((2,2), strides=(2,2))(conv2)

    conv3 = Convolution2D(128, 3, 3, border_mode='same', trainable = train_all)(pool2)
    conv3 = ELU()(conv3)
    conv3 = BatchNormalization(axis=1, mode=0)(conv3)
    #conv3 = Dropout(0.1)(conv3)
    conv3 = Convolution2D(128, 3, 3, border_mode='same', trainable = train_all)(conv3)
    conv3 = ELU()(conv3)
    conv3 = BatchNormalization(axis=1, mode=0)(conv3)
    #conv3 = Dropout(0.1)(conv3)
    conv3 = Convolution2D(128, 3, 3, border_mode='same', trainable = train_all)(conv3)
    conv3 = ELU()(conv3)
    conv3 = BatchNormalization(axis=1, mode=0)(conv3)

    pool3 = MaxPooling2D((2,2), strides=(2,2))(conv3)

    conv4 = Convolution2D(256, 3, 3, border_mode='same', trainable = train_all)(pool3)
    conv4 = ELU()(conv4)
    conv4 = BatchNormalization(axis=1, mode=0)(conv4)
    #conv4 = Dropout(0.1)(conv4)
    conv4 = Convolution2D(256, 3, 3, border_mode='same', trainable = train_all)(conv4)
    conv4 = ELU()(conv4)
    conv4 = BatchNormalization(axis=1, mode=0)(conv4)
    #conv4 = Dropout(0.1)(conv4)
    conv4 = Convolution2D(256, 3, 3, border_mode='same', trainable = train_all)(conv4)
    conv4 = ELU()(conv4)
    conv4 = BatchNormalization(axis=1, mode=0)(conv4)

    pool4 = MaxPooling2D((2,2), strides=(2,2))(conv4)

    conv5 = Convolution2D(512, 3, 3, border_mode='same', trainable = train_all)(pool4)
    conv5 = ELU()(conv5)
    conv5 = BatchNormalization(axis=1, mode=0)(conv5)
    #conv5 = Dropout(0.1)(conv5)
    conv5 = Convolution2D(512, 3, 3, border_mode='same', trainable = train_all)(conv5)
    conv5 = ELU()(conv5)
    conv5 = BatchNormalization(axis=1, mode=0)(conv5)
    #conv5 = Dropout(0.1)(conv5)
    conv5 = Convolution2D(512, 3, 3, border_mode='same', trainable = train_all)(conv5)
    conv5 = ELU()(conv5)
    conv5 = BatchNormalization(axis=1, mode=0)(conv5)

    pool5 = MaxPooling2D((2,2), strides=(2,2))(conv5)

    conv6 = Convolution2D(512, 3, 3, border_mode='same', trainable = train_all)(pool5)
    conv16= ELU()(conv6)
    conv6 = BatchNormalization(axis=1, mode=0)(conv6)
    #conv6 = Dropout(0.1)(conv6)
    conv6 = Convolution2D(512, 3, 3, border_mode='same', trainable = train_all)(conv6)
    conv6 = ELU()(conv6)
    conv6 = BatchNormalization(axis=1, mode=0)(conv6)


    up8 = UpSampling2D((8,8))(conv6)
    up8 = Convolution2D(256, 3, 3, border_mode='same', )(up8)
    up8 = ELU()(up8)
    up8 = BatchNormalization(axis=1, mode=0)(up8)
    #up8 = Dropout(0.5)(up8)
    up8 = Convolution2D(256, 3, 3, border_mode='same', )(up8)
    up8 = ELU()(up8)
    up8 = BatchNormalization(axis=1, mode=0)(up8)    

    up4 = UpSampling2D((4,4))(pool4)
    up4 = Convolution2D(256, 3, 3, border_mode='same', )(up4)
    up4 = ELU()(up4)
    up4 = BatchNormalization(axis=1, mode=0)(up4)
    #up4 = Dropout(0.5)(up4)
    up4 = Convolution2D(256, 3, 3, border_mode='same', )(up4)
    up4 = ELU()(up4)
    up4 = BatchNormalization(axis=1, mode=0)(up4)

    up2 = UpSampling2D((2,2))(pool3)
    up2 = Convolution2D(128, 3, 3, border_mode='same', )(up2)
    up2 = ELU()(up2)
    up2 = BatchNormalization(axis=1, mode=0)(up2)
    #up2 = Dropout(0.5)(up2)
    up2 = Convolution2D(128, 3, 3, border_mode='same', )(up2)
    up2 = ELU()(up2)
    up2 = BatchNormalization(axis=1, mode=0)(up2)

    up1 = Convolution2D(64, 3, 3, border_mode='same', )(pool2)
    up1 = ELU()(up1)
    up1 = BatchNormalization(axis=1, mode=0)(up1)
    #up1 = Dropout(0.5)(up1)
    up1 = Convolution2D(64, 3, 3, border_mode='same', )(up1)
    up1 = ELU()(up1)
    up1 = BatchNormalization(axis=1, mode=0)(up1)

    merge_up = merge([up1, up2, up4, up8], mode='concat', concat_axis=1, output_shape=(704,24,32))


    classify = Convolution2D(256, 3, 3, border_mode='same')(merge_up)
    classify = ELU()(classify)
    classify = BatchNormalization(axis=1, mode=0)(classify)
    classify = Dropout(0.2)(classify)
    classify = Convolution2D(64, 3, 3, border_mode='same')(merge_up)
    classify = ELU()(classify)
    classify = BatchNormalization(axis=1, mode=0)(classify)
    classify = Dropout(0.2)(classify)
    classify = Convolution2D(16, 3, 3, border_mode='same')(classify)
    classify = ELU()(classify)
    classify = BatchNormalization(axis=1, mode=0)(classify)

    outputs = UpSampling2D((4,4))(classify)

    outputs = Convolution2D(8, 8, 8, border_mode = 'same')(outputs)
    outputs = ELU()(outputs)
    outputs = BatchNormalization(axis=1, mode=0)(outputs)
    #outputs = Dropout(0.2)(outputs)
    outputs = Convolution2D(8, 8, 8, border_mode = 'same')(outputs)
    outputs = ELU()(outputs)
    outputs = BatchNormalization(axis=1, mode=0)(outputs)
    #outputs = Dropout(0.2)(outputs)
    outputs = Convolution2D(1, 1, 1, activation='sigmoid')(outputs)

    model = Model(input=inputs, output=outputs)

    if weights_path:
        model.load_weights(weights_path)
        
    sgd = SGD(lr=0.1, momentum=0.9, decay=0.001, nesterov=False)
    adam = Adam(lr=.01)
    model.compile(optimizer=sgd, loss = dice_coef_loss, metrics=[dice_coef])
    
    return model


#an earlier iteration of the model, that was used to train the classifier (was not enough time to retrain with the newer model)

def FCN1_model(weights_path=None, train_all = True):


    inputs = Input(shape=(1,96,128))

    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', trainable = train_all)(inputs)
    conv1 = BatchNormalization(axis=1, mode=2)(conv1)
    #conv1 = Dropout(0.1)(conv1)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', trainable = train_all)(conv1)
    conv1 = BatchNormalization(axis=1, mode=2)(conv1)

    pool1 = MaxPooling2D((2,2), strides=(2,2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', trainable = train_all)(pool1)
    conv2 = BatchNormalization(axis=1, mode=2)(conv2)
    #conv2 = Dropout(0.1)(conv2)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', trainable = train_all)(conv2)
    conv2 = BatchNormalization(axis=1, mode=2)(conv2)

    pool2 = MaxPooling2D((2,2), strides=(2,2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', trainable = train_all)(pool2)
    conv3 = BatchNormalization(axis=1, mode=2)(conv3)
    #conv3 = Dropout(0.1)(conv3)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', trainable = train_all)(conv3)
    conv3 = BatchNormalization(axis=1, mode=2)(conv3)
    #conv3 = Dropout(0.1)(conv3)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', trainable = train_all)(conv3)
    conv3 = BatchNormalization(axis=1, mode=2)(conv3)

    pool3 = MaxPooling2D((2,2), strides=(2,2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', trainable = train_all)(pool3)
    conv4 = BatchNormalization(axis=1, mode=2)(conv4)
    #conv4 = Dropout(0.1)(conv4)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', trainable = train_all)(conv4)
    conv4 = BatchNormalization(axis=1, mode=2)(conv4)
    #conv4 = Dropout(0.1)(conv4)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', trainable = train_all)(conv4)
    conv4 = BatchNormalization(axis=1, mode=2)(conv4)

    pool4 = MaxPooling2D((2,2), strides=(2,2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', trainable = train_all)(pool4)
    conv5 = BatchNormalization(axis=1, mode=2)(conv5)
    #conv5 = Dropout(0.1)(conv5)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', trainable = train_all)(conv5)
    conv5 = BatchNormalization(axis=1, mode=2)(conv5)
    #conv5 = Dropout(0.1)(conv5)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', trainable = train_all)(conv5)
    conv5 = BatchNormalization(axis=1, mode=2)(conv5)

    pool5 = MaxPooling2D((2,2), strides=(2,2))(conv5)

    conv6 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', trainable = train_all)(pool5)
    conv6 = BatchNormalization(axis=1, mode=2)(conv6)
    #conv6 = Dropout(0.1)(conv6)
    conv6 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', trainable = train_all)(conv6)
    conv6 = BatchNormalization(axis=1, mode=2)(conv6)


    up8 = UpSampling2D((8,8))(conv6)
    up8 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', )(up8)
    up8 = BatchNormalization(axis=1, mode=2)(up8)
    #up4 = Dropout(0.1)(up4)
    up8 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', )(up8)
    up8 = BatchNormalization(axis=1, mode=2)(up8)    

    up4 = UpSampling2D((4,4))(pool4)
    up4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', )(up4)
    up4 = BatchNormalization(axis=1, mode=2)(up4)
    #up4 = Dropout(0.1)(up4)
    up4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', )(up4)
    up4 = BatchNormalization(axis=1, mode=2)(up4)

    up2 = UpSampling2D((2,2))(pool3)
    up2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', )(up2)
    up2 = BatchNormalization(axis=1, mode=2)(up2)
    #up2 = Dropout(0.1)(up2)
    up2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', )(up2)
    up2 = BatchNormalization(axis=1, mode=2)(up2)

    up1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', )(pool2)
    up1 = BatchNormalization(axis=1, mode=2)(up1)
    #up1 = Dropout(0.1)(up1)
    up1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', )(up1)
    up1 = BatchNormalization(axis=1, mode=2)(up1)

    merge_up = merge([up1, up2, up4, up8], mode='concat', concat_axis=1, output_shape=(704,24,32))


    classify = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(merge_up)
    classify = BatchNormalization(axis=1, mode=2)(classify)
    #classify = Dropout(0.1)(classify)
    classify = Convolution2D(1, 1, 1, activation='sigmoid')(classify)

    outputs = UpSampling2D((4,4))(classify)

    model = Model(input=inputs, output=outputs)

    if weights_path:
        model.load_weights(weights_path)
        
    sgd = SGD(lr=0.1, momentum=0.9, decay=0.001, nesterov=False)
    model.compile(optimizer=sgd, loss = dice_coef_loss, metrics=[dice_coef])
    
    return model


#a simple conv-net based on VGG-style architecture, used to classify presence of nerve
#trained with weight regularization (dropout was counter-productive in CV)
def simple_classifier(weights_path=None):

    inputs = Input(shape=(1,96, 128))


    conv1 = Convolution2D(16, 3, 3, activation='relu', border_mode='same',)(inputs)
    conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same',)(conv1)

    pool1 = MaxPooling2D((2,2), strides=(2,2))(conv2)

    conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same',)(pool1)
    conv4 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', )(conv3)

    pool2 = MaxPooling2D((2,2), strides=(2,2))(conv4)

    conv5 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', )(pool2)
    conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', )(conv5)

    pool3 = MaxPooling2D((2,2), strides=(2,2))(conv6)

    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same',)(pool3)
    conv8 = Convolution2D(128, 3, 3, activation='relu', border_mode='same',)(conv7)

    outputs = Convolution2D(1, 12, 16, activation='sigmoid', W_regularizer = l1l2(l1=0.0001, l2=0.1))(conv8)
    outputs = Flatten()(outputs)

    model = Model(input=inputs, output=outputs)

    if weights_path:
        model.load_weights(weights_path)
        
    ad = Adam(lr=0.001)
    model.compile(optimizer=ad, loss = 'binary_crossentropy', metrics=['accuracy'])

    return model 


#classifier using the FCN model with weights from its training
#can either retrain the whole network or just the layers after upsampling

def FCN_class(weights_path=None, u_weights = None, trainable = False):

    inputs = Input(shape=(1,96,128), name='inputs')

    fcn_net = FCN1_model(u_weights, train_all = trainable)

    fcn_out = fcn_net(inputs)

    conv1 = Convolution2D(16, 3, 3, activation='relu', border_mode='same',)(u_out)
    conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same',)(conv1)

    pool1 = MaxPooling2D((2,2), strides=(2,2))(conv2)

    conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same',)(pool1)
    conv4 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', )(conv3)

    pool2 = MaxPooling2D((2,2), strides=(2,2))(conv4)

    conv5 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', )(pool2)
    conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', )(conv5)

    pool3 = MaxPooling2D((2,2), strides=(2,2))(conv6)

    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same',)(pool3)
    conv8 = Convolution2D(128, 3, 3, activation='relu', border_mode='same',)(conv7)


    outputs = Convolution2D(1, 12, 16, activation='sigmoid', W_regularizer = l1l2(l1=0.0001, l2=0.1))(conv8)
    outputs = Flatten()(outputs)

    model = Model(input= inputs, output= outputs)

    if weights_path:
        model.load_weights(weights_path)
        
    sgd = SGD(lr=0.1, momentum=0.9, decay=0.001, nesterov=False)
    adam = Adam(lr=.001)
    model.compile(optimizer=adam, loss = 'binary_crossentropy', 
                metrics=['accuracy'])
    
    return model


