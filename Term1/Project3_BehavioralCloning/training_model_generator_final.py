#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 13:48:58 2016

@author: amit
"""
from sklearn.model_selection import train_test_split
from training_data_generator_final import *
###from training_data_generation_generator_randomized_2_prob_areas import *
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

#load validation data (from Udacity's data)
X_val, y_val = generate_validation_data()

#Model definition

#constants
batch_size = 32 #8 #its really 64*3*4-1 (you can tell by the sample update as its training the model)
num_samples_per_epoch = (8036-256)*4
nb_epochs = 16
features_input_shape = generate_data_generator(batch_size=batch_size).__next__()[0].shape[1:] #for generator version
conv_filter_size = (3,3)
conv_stride = (1,1)
pool_filter_size = (2,2)
pool_stride = pool_filter_size #cuts input size in half (by the size of the filter)
dropout_prob = 0.5
learning_rate = 0.001 #0.001/25 #0.00001 #0.001 is default for Adam
l2_reg = 1e-6 #0.0001

nb_filter_conv2 = 24
nb_filter_conv3 = 36
nb_filter_conv4 = 48
nb_filter_conv5 = 64
nb_filter_conv6 = 64

nb_fc7 = 100
nb_fc8 = 50
nb_fc9 = 10
nb_prediction = 1 #single output regression problem

#inputs layer's output is (?x80x160x3)
inputs = Input(shape=features_input_shape) #this returns a tensor
#Input normalization layer's output is (?x80x160x3)
inputs_normalized = Lambda(lambda xin: (xin/255-0.5)*2)(inputs) #normalize the input
#Layer1 output is (?x80x160x3) 
#layer1 is to train the network to learn which is the right color space instead of explicitly telling it whether it is RGB, HSV,etc
conv1 = Convolution2D(nb_filter=3, nb_row=1, nb_col=1, border_mode='same', subsample=(1,1),
                      init='glorot_normal', W_regularizer=l2(l2_reg))(inputs_normalized)
conv1 = Activation('elu')(conv1)
#Layer2 output is (?x40x80x24)
conv2 = Convolution2D(nb_filter=nb_filter_conv2, nb_row=conv_filter_size[0], nb_col=conv_filter_size[1],
                      border_mode='same', subsample=conv_stride, init='glorot_normal', W_regularizer = l2(l2_reg))(conv1)  #Glorot aka Xavier method of initialization
conv2 = MaxPooling2D(pool_size=pool_filter_size, strides=pool_stride)(conv2)
conv2 = Activation('elu')(conv2) #activation after maxpooling is faster computationally
conv2 = Dropout(dropout_prob)(conv2)
#Layer3 output is (?x20x40x36)
conv3 = Convolution2D(nb_filter=nb_filter_conv3, nb_row=conv_filter_size[0], nb_col=conv_filter_size[1],
                      border_mode='same', subsample=conv_stride, init='glorot_normal', W_regularizer = l2(l2_reg))(conv2)
conv3 = MaxPooling2D(pool_size=pool_filter_size, strides=pool_stride)(conv3)
conv3 = Activation('elu')(conv3)
conv3 = Dropout(dropout_prob)(conv3)
#Layer4 output is (?x10x20x48)
conv4 = Convolution2D(nb_filter=nb_filter_conv4, nb_row=conv_filter_size[0], nb_col=conv_filter_size[1],
                      border_mode='same', subsample=conv_stride, init='glorot_normal', W_regularizer = l2(l2_reg))(conv3)
conv4 = MaxPooling2D(pool_size=pool_filter_size, strides=pool_stride)(conv4)
conv4 = Activation('elu')(conv4)
conv4 = Dropout(dropout_prob)(conv4)
#Layer5 output is (?x5x10x64)
conv5 = Convolution2D(nb_filter=nb_filter_conv5, nb_row=conv_filter_size[0], nb_col=conv_filter_size[1],
                      border_mode='same', subsample=conv_stride, init='glorot_normal', W_regularizer = l2(l2_reg))(conv4)
conv5 = MaxPooling2D(pool_size=pool_filter_size, strides=pool_stride)(conv5)
conv5 = Activation('elu')(conv5)
conv5 = Dropout(dropout_prob)(conv5)
#Layer6 output is (?x5x10x64)
conv6 = Convolution2D(nb_filter=nb_filter_conv6, nb_row=conv_filter_size[0], nb_col=conv_filter_size[1],
                      border_mode='same', subsample=conv_stride, init='glorot_normal', W_regularizer = l2(l2_reg))(conv5)
###conv6 = MaxPooling2D(pool_size=pool_filter_size, strides=pool_stride)(conv6) #no maxpooling
conv6 = Activation('elu')(conv6)
conv6 = Dropout(dropout_prob)(conv6)
#flatten before going to fully connected (5*10*64=3200 features)(?x3200)
conv6_flatten = Flatten()(conv6)
#Layer7 output is (?x100)
fc7 = Dense(output_dim=nb_fc7, activation='elu', W_regularizer = l2(l2_reg))(conv6_flatten)
fc7 = Dropout(dropout_prob)(fc7)
#Layer8 output is (?x50)
fc8 = Dense(output_dim=nb_fc8, activation='elu', W_regularizer = l2(l2_reg))(fc7)
fc8 = Dropout(dropout_prob)(fc8)
#Layer9 output is (?x10)
fc9 = Dense(output_dim=nb_fc8, activation='elu', W_regularizer = l2(l2_reg))(fc8)
fc9 = Dropout(dropout_prob)(fc9)
#Output layer's output is (?x1)
prediction = Dense(output_dim=nb_prediction)(fc9) #no activation or dropout on the output layer
# and no softmax as its a regression problem and not classification

model = Model(input=inputs, output=prediction)
print(model.summary())

adam = Adam(lr=learning_rate)
model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
#model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

################# Generator version#########################

model.load_weights('test_models/final_working_best/coarse_13 is best/model_13.h5') #, by_name=True) #load previously save weights. don't do by_name=True for identical model. It doesn't laod the weights as the initial loss is still high doing so.

import json
for i_epoch in range(nb_epochs):
    str_rej = 0.4 #0.2 + 0.7/(1.0+i_epoch)
    print('Epoch number: {0}'.format(i_epoch))
    history = model.fit_generator(generate_data_generator(batch_size=batch_size, straight_steer_discard_prob_th=str_rej),
                                  samples_per_epoch=num_samples_per_epoch, nb_epoch=1, verbose=1, validation_data=(X_val,y_val))

    #saving the model architecture as json file and the weights as hdf5 file
    json_string = model.to_json()
    file_name = 'test_models/model_FT'+str(i_epoch)+'.json'
    with open(file_name, 'w') as f:
        json.dump(json_string,f)
    #saving the weights
    file_name = 'test_models/model_FT'+str(i_epoch)+'.h5'
    model.save_weights(file_name)

##restoring the json model
#with open('model.json') as f:
#    json_string = json.load(f)
#    
##model reconstruciton from JSON
#from keras.models import model_from_json
#model = model_from_json(json_string)
#
##load the weights
##model.load_weights('model.h5') #restore all the weight for an identical model
#model.load_weights('model.h5', by_name=True) #restores weights for same layers (good for fine tuning on a existing but slightly different model)

#final time
tf = time.time()
print('Time to train: {0:.2}seconds'.format(tf-to))