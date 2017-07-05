#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 13:48:58 2016

@author: amit
"""
from sklearn.model_selection import train_test_split
from training_data_generation_generator_randomized import *
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
batch_size = 64 #8 #its really 64*3*4-1 (you can tell by the sample update as its training the model)
num_samples_per_epoch = 8036*2 #3782*3*4 #(24108+330)*4 #total number of training images (left,right,center)*4 (normal,brightness,translate,flip)
nb_epochs = 16
features_input_shape = generate_data_generator(batch_size=batch_size).__next__()[0].shape[1:] #for generator version

learning_rate = 0.001 #0.001 is default for Adam

#inputs layer's output is (?x80x160x3)
inputs = Input(shape=features_input_shape) #this returns a tensor
#Input normalization layer's output is (?x80x160x3)
inputs_normalized = Lambda(lambda xin: (xin/255-0.5)*2)(inputs) #normalize the input

####comma.ai's model##############
#conv1 output is (?x80x160x16)
conv1 = Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same")(inputs_normalized)
conv1 = Activation('elu')(conv1)
#conv2 output is (?x80x160x32)
conv2 = Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same")(conv1)
conv2 = Activation('elu')(conv2)
#conv3 output is (?x)
conv3 = Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same")(conv2)
conv3_flt = Flatten()(conv3)
conv3_flt = Dropout(.2)(conv3_flt)
conv3_flt = Activation('elu')(conv3_flt)
fc4 = Dense(512)(conv3_flt)
fc4 = Dropout(.5)(fc4)
fc4 = Activation('elu')(fc4)
prediction = Dense(1)(fc4)
##########################
model = Model(input=inputs, output=prediction)
print(model.summary())

adam = Adam(lr=learning_rate)
model.compile(optimizer=adam, loss='mse')

################# Generator version#########################

#model.load_weights('test_models/model_15.h5') #, by_name=True) #load previously save weights. don't do by_name=True for identical model. It doesn't laod the weights as the initial loss is still high doing so.

import json
for i_epoch in range(nb_epochs):
    history = model.fit_generator(generate_data_generator(batch_size=batch_size), samples_per_epoch=num_samples_per_epoch, 
                              nb_epoch=1, verbose=1, validation_data=None)

    #saving the model architecture as json file and the weights as hdf5 file
    json_string = model.to_json()
    file_name = 'test_models/model_'+str(i_epoch)+'.json'
    with open(file_name, 'w') as f:
        json.dump(json_string,f)
    #saving the weights
    file_name = 'test_models/model_'+str(i_epoch)+'.h5'
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