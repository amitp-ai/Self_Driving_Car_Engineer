#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 13:48:58 2016

@author: amit
"""
from sklearn.model_selection import train_test_split
from training_data_generation_generator import *
import matplotlib.pyplot as plt
import numpy as np
import time

from keras.layers import Input, Dense, Lambda, Flatten, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

#initial time
to = time.time()

#Model definition

#constants
batch_size = 128 #its really 64*3-1 (you can tell by the sample update as its training the model)
num_samples_per_epoch = 24108 #total number of training images (left,right,center)
nb_epochs = 5
features_input_shape = generate_data_generator(batch_size=batch_size).__next__()[0].shape[1:] #for generator version
conv_filter_size = (3,3)
conv_stride = (1,1)
pool_filter_size = (2,2)
pool_stride = pool_filter_size #cuts input size in half (actually by the size of the filter)
dropout_prob = 0.5
learning_rate = 0.001 #0.001 is default for Adam
l2_reg = 0.0001

nb_filter_conv2 = 16
nb_filter_conv3 = 16
nb_filter_conv4 = 32
nb_filter_conv5 = 32
nb_filter_conv6 = 64
nb_filter_conv7 = 64

nb_fc8 = 128
nb_fc9 = 64
nb_fc10 = 8
nb_prediction = 1 #single output regression problem

#inputs layer's output is (?x80x160x3)
inputs = Input(shape=features_input_shape) #this returns a tensor
#Input normalization layer's output is (?x80x160x3)
inputs_normalized = Lambda(lambda xin: (xin/255-0.5)*2)(inputs) #normalize the input
#Layer1 output is (?x80x160x3) 
#layer1 is to train the network to learn which is the right color space instead of explicitly telling it whether it is RGB, HSV,etc
conv1 = Convolution2D(nb_filter=3, nb_row=1, nb_col=1, border_mode='valid', subsample=(1,1),
                      init='glorot_normal', W_regularizer=l2(l2_reg))(inputs_normalized)
#Layer2 output is (?x80x160x16)
conv2 = Convolution2D(nb_filter=nb_filter_conv2, nb_row=conv_filter_size[0], nb_col=conv_filter_size[1],
                      border_mode='same', subsample=conv_stride, init='glorot_normal', W_regularizer = l2(l2_reg))(conv1)  #Glorot aka Xavier method of initialization
#Layer3 output is (?x80x160x16)
conv3 = Convolution2D(nb_filter=nb_filter_conv3, nb_row=conv_filter_size[0], nb_col=conv_filter_size[1],
                      border_mode='same', subsample=conv_stride, init='glorot_normal', W_regularizer = l2(l2_reg))(conv2)  #Glorot aka Xavier method of initialization
#Maxpool/Activation/Dropout layer 3 output is (?x40x80x16)
conv3 = MaxPooling2D(pool_size=pool_filter_size, strides=pool_stride)(conv3)
conv3 = Activation('elu')(conv3) #activation after maxpooling is faster computationally
conv3 = Dropout(dropout_prob)(conv3)
#Layer4 (?x40x80x32)
conv4 = Convolution2D(nb_filter=nb_filter_conv4, nb_row=conv_filter_size[0], nb_col=conv_filter_size[1],
                      border_mode='same', subsample=conv_stride, init='glorot_normal', W_regularizer = l2(l2_reg))(conv3)
#Layer5 output is (?x40x80x32)
conv5 = Convolution2D(nb_filter=nb_filter_conv5, nb_row=conv_filter_size[0], nb_col=conv_filter_size[1],
                      border_mode='same', subsample=conv_stride, init='glorot_normal', W_regularizer = l2(l2_reg))(conv4)
#Maxpool/Activation/Dropout layer 5 output is (?x20x40x32)
conv5 = MaxPooling2D(pool_size=pool_filter_size, strides=pool_stride)(conv5)
conv5 = Activation('elu')(conv5)
conv5 = Dropout(dropout_prob)(conv5)
#Layer6 (?x20x40x64)
conv6 = Convolution2D(nb_filter=nb_filter_conv6, nb_row=conv_filter_size[0], nb_col=conv_filter_size[1],
                      border_mode='same', subsample=conv_stride, init='glorot_normal', W_regularizer = l2(l2_reg))(conv5)
#Layer7 output is (?x20x40x64)
conv7 = Convolution2D(nb_filter=nb_filter_conv7, nb_row=conv_filter_size[0], nb_col=conv_filter_size[1],
                      border_mode='same', subsample=conv_stride, init='glorot_normal', W_regularizer = l2(l2_reg))(conv6)
#Maxpool/Activation/Dropout layer 7 (?x5x10x64)
conv7 = MaxPooling2D(pool_size=(4,4), strides=(4,4))(conv7) #maxpooling to reduce the size by 4X
conv7 = Activation('elu')(conv7)
conv7 = Dropout(dropout_prob)(conv7)
#flatten before going to fully connected (5*10*64=3200 features=> ?x3200)
conv7_flatten = Flatten()(conv7)
#Layer8 output is (?x128)
fc8 = Dense(output_dim=nb_fc8, activation='elu', W_regularizer = l2(l2_reg))(conv7_flatten)
fc8 = Dropout(dropout_prob)(fc8)
#Layer9 output is (?x64)
fc9 = Dense(output_dim=nb_fc9, activation='elu', W_regularizer = l2(l2_reg))(fc8)
fc9 = Dropout(dropout_prob)(fc9)
#Layer10 output is (?x8)
fc10 = Dense(output_dim=nb_fc10, activation='elu', W_regularizer = l2(l2_reg))(fc9)
fc10 = Dropout(dropout_prob)(fc10)
#Output layer (?x1)
prediction = Dense(output_dim=nb_prediction)(fc10) #no activation or dropout on the output layer
# and no softmax as its a regression problem and not classification

model = Model(input=inputs, output=prediction)
print(model.summary())

adam = Adam(lr=learning_rate)
model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
#model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

################# Generator version#########################
#model.load_weights('model.h5') #, by_name=True) #load previously saved weights. don't do by_name=True for identical model. It doesn't laod the weights as the initial loss is still high doing so.

history = model.fit_generator(generate_data_generator(batch_size=batch_size), samples_per_epoch=num_samples_per_epoch, 
                              nb_epoch=nb_epochs, verbose=1, validation_data=None)

final_train_acc = history.history['acc'][-1]
print('Final epochs training accuracy: ', final_train_acc) #final epoch's validation accuracy

score = model.evaluate_generator(generate_data_generator(batch_size=128), val_samples = 1) #evaluate gives the loss and accuracy (if using accuracy metric in model.compile)
print(score)
print(model.metrics_names)

#predict_generator doesn't work gives error
#pred = model.predict_generator(generate_data_generator().__next__()[0:1][0], val_samples = 1)[0]
#print('prediciton:', pred) #to keep it 4D tensor for a single example

###############################################################

#saving the model architecture as json file and the weights as hdf5 file
import json
json_string = model.to_json()
with open('model.json', 'w') as f:
    json.dump(json_string,f)

#saving the weights
model.save_weights('model.h5')

del model #deletes the existing model

#restoring the json model
with open('model.json') as f:
    json_string = json.load(f)
    
#model reconstruciton from JSON
from keras.models import model_from_json
model = model_from_json(json_string)

#load the weights
#model.load_weights('model.h5') #restore all the weight for an identical model
model.load_weights('model.h5', by_name=True) #restores weights for same layers (good for fine tuning on a existing but slightly different model)

#predict_generator doesn't work gives error. see above.
#pred = model.predict_generator(features_val[0:1])[0]
#print('reloaded models prediciton:', pred) #to keep it 4D tensor for a single example

#final time
tf = time.time()
print('Time to train: {0:.2}seconds'.format(tf-to))