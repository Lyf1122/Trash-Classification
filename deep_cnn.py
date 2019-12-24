# -*- coding:utf-8 -*-
from __future__ import print_function, division
import os
import numpy as np
import keras.backend as K
import tensorflow as tf
from keras import optimizers, losses
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import to_categorical

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# datasets for training and validation
set3 = np.load('/home/ubuntu/MyFiles/lyf/data/set3_48_data.npy')
set3_label = np.load('/home/ubuntu/MyFiles/lyf/data/set3_48_label.npy')
set3_label = np.reshape(a=set3_label, newshape=(len(set3_label)))
set3_label_c = to_categorical(y=set3_label, num_classes=2)

set4 = np.load('/home/ubuntu/MyFiles/lyf/data/set4_48_data.npy')
set4_label = np.load('/home/ubuntu/MyFiles/lyf/data/set4_48_label.npy')
set4_label = np.reshape(a=set4_label, newshape=(len(set4_label)))
set4_label_c = to_categorical(y=set4_label, num_classes=2)

train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()
train_data = train_datagen.flow(
    x=set3,
    y=set3_label_c,
    batch_size=96,
    shuffle=True)
val_data = val_datagen.flow(x=set4, y=set4_label_c, batch_size=100)

# metrics for training


def tf_count(t, val):
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)
    return count


def loss(y_true, y_pred):

    l = losses.categorical_crossentropy(y_pred=y_pred, y_true=y_true)
    los = l

    return los

# build model


#sgd = optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-5)
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=1e-5)
model_DNCNN = Sequential()
model_DNCNN.add(
    Convolution2D(
        nb_filter=32,
        nb_col=3,
        nb_row=3,
        subsample=(
            1,
            1),
        border_mode='same',
        name='CONV1_1',
        input_shape=(
            48,
            48,
            3)))
# model_DNCNN.add(BatchNormalization(name='BN1_1'))
model_DNCNN.add(Activation('relu'))
model_DNCNN.add(
    Convolution2D(
        nb_filter=32,
        nb_col=3,
        nb_row=3,
        subsample=(
            1,
            1),
        border_mode='same',
        name='CONV1_2'))
# model_DNCNN.add(BatchNormalization(name='BN1_2'))
model_DNCNN.add(Activation('relu'))
model_DNCNN.add(
    Convolution2D(
        nb_filter=64,
        nb_col=3,
        nb_row=3,
        subsample=(
            1,
            1),
        border_mode='same',
        name='CONV1_3'))
# model_DNCNN.add(BatchNormalization(name='BN1_3'))
model_DNCNN.add(Activation('relu'))
model_DNCNN.add(
    MaxPooling2D(
        pool_size=(
            3, 3), strides=(
                2, 2), border_mode='same', name='pool1'))

model_DNCNN.add(
    Convolution2D(
        nb_filter=64,
        nb_col=3,
        nb_row=3,
        subsample=(
            1,
            1),
        border_mode='same',
        name='CONV2_1'))
# model_DNCNN.add(BatchNormalization(name='BN2_1'))
model_DNCNN.add(Activation('relu'))
model_DNCNN.add(
    Convolution2D(
        nb_filter=64,
        nb_col=3,
        nb_row=3,
        subsample=(
            1,
            1),
        border_mode='same',
        name='CONV2_2'))
# model_DNCNN.add(BatchNormalization(name='BN2_2'))
model_DNCNN.add(Activation('relu'))
model_DNCNN.add(
    MaxPooling2D(
        pool_size=(
            2, 2), strides=(
                2, 2), border_mode='valid', name='pool2'))

model_DNCNN.add(
    Convolution2D(
        nb_filter=384,
        nb_col=3,
        nb_row=3,
        subsample=(
            1,
            1),
        border_mode='same',
        name='CONV3'))
# model_DNCNN.add(BatchNormalization(name='BN3'))
model_DNCNN.add(Activation('relu'))
model_DNCNN.add(
    Convolution2D(
        nb_filter=384,
        nb_col=3,
        nb_row=3,
        subsample=(
            1,
            1),
        border_mode='same',
        name='CONV4'))
# model_DNCNN.add(BatchNormalization(name='BN4'))
model_DNCNN.add(Activation('relu'))
model_DNCNN.add(
    Convolution2D(
        nb_filter=256,
        nb_col=3,
        nb_row=3,
        subsample=(
            1,
            1),
        border_mode='same',
        name='CONV5'))
# model_DNCNN.add(BatchNormalization(name='BN5'))
model_DNCNN.add(Activation('relu'))
model_DNCNN.add(
    MaxPooling2D(
        pool_size=(
            2, 2), strides=(
                2, 2), border_mode='valid', name='pool3'))

model_DNCNN.add(Flatten())
model_DNCNN.add(Dense(output_dim=2048, activation='relu', name='FC1'))
model_DNCNN.add(Dropout(rate=0.5))
model_DNCNN.add(Dense(output_dim=2048, activation='relu', name='FC2'))
model_DNCNN.add(Dropout(rate=0.5))
model_DNCNN.add(Dense(output_dim=2, activation='softmax', name='OUT'))
model_DNCNN.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
print('The model was created and compiled!')


print('Now start to training......')
tensorboard = TensorBoard(log_dir='../tensor_log/DNCNN_mode')
modelCheckpoint = ModelCheckpoint(
    filepath='../model/DNCNN.hdf5',
    monitor='val_acc',
    mode='max',
    save_best_only=True)
model_DNCNN.fit_generator(
    generator=train_data,
    validation_data=val_data,
    nb_epoch=300,
    verbose=2,
    steps_per_epoch=167,
    validation_steps=165,
    callbacks=[
        modelCheckpoint,
        tensorboard])
print('Training was completed!')
