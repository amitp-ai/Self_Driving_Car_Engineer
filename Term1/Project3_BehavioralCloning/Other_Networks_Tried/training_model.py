#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 13:48:58 2016

@author: amit
"""
from sklearn.model_selection import train_test_split
from training_data_generation import *
import matplotlib.pyplot as plt
import numpy as np
import time


from keras.layers import Input, Dense, Lambda, Flatten, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model

to = time.time()

################ Non-generator version#########################
#load data
###Don't load the file. Its too big
###data_file = 'camera_training_data.p'
##with open(data_file, 'rb') as f: #read in binary format
##    camera_training_data = pickle.load(f)

features = generate_data()['features'] #for non-generator version
steering_angle = generate_data()['steering_angle'] #for non-generator version

#randomly shuffle the training data (for non-generator version)
features_train, features_val, steering_angle_train, steering_angle_val = train_test_split(features, steering_angle, test_size=0.05, random_state=0)

#features=features.astype(np.uint8)  #note b'cse features is float32 is doesn't display properly. convert it to uint8 to display properly.
#plt.imshow(features[0])

#################################################################

#Model definition

#constants
batch_size = 512
nb_epochs = 10
features_input_shape = features.shape[1:] #for non-generator version
conv_filter_size = (3,3)
conv_stride = (1,1)
pool_filter_size = (2,2)
pool_stride = pool_filter_size #cuts input size in half (by the size of the filter)
dropout_prob = 0.5

nb_filter_conv2 = 24
nb_filter_conv3 = 36
nb_filter_conv4 = 48
nb_filter_conv5 = 64
nb_filter_conv6 = 64

nb_fc7 = 100
nb_fc8 = 50
nb_fc9 = 10
nb_prediction = 1 #single output regression problem

#inputs (80x160x3)
inputs = Input(shape=features_input_shape) #this returns a tensor
#layer 1 (80x160x3)
inputs_normalized1 = Lambda(lambda xin: (xin/255-0.5)*2)(inputs) #normalize the input
#Layer2 (40x80x24)
conv2 = Convolution2D(nb_filter=nb_filter_conv2, nb_row=conv_filter_size[0], nb_col=conv_filter_size[1],
                      border_mode='same', subsample=conv_stride, init='glorot_normal')(inputs_normalized1)  #Glorot aka Xavier method of initialization
conv2 = MaxPooling2D(pool_size=pool_filter_size, strides=pool_stride)(conv2)
conv2 = Activation('elu')(conv2) #activation after maxpooling is faster computationally
conv2 = Dropout(dropout_prob)(conv2)
#Layer3 (20x40x36)
conv3 = Convolution2D(nb_filter=nb_filter_conv3, nb_row=conv_filter_size[0], nb_col=conv_filter_size[1],
                      border_mode='same', subsample=conv_stride, init='glorot_normal')(conv2)
conv3 = MaxPooling2D(pool_size=pool_filter_size, strides=pool_stride)(conv3)
conv3 = Activation('elu')(conv3)
conv3 = Dropout(dropout_prob)(conv3)
#Layer4 (10x20x48)
conv4 = Convolution2D(nb_filter=nb_filter_conv4, nb_row=conv_filter_size[0], nb_col=conv_filter_size[1],
                      border_mode='same', subsample=conv_stride, init='glorot_normal')(conv3)
conv4 = MaxPooling2D(pool_size=pool_filter_size, strides=pool_stride)(conv4)
conv4 = Activation('elu')(conv4)
conv4 = Dropout(dropout_prob)(conv4)
#Layer5 (5x10x64)
conv5 = Convolution2D(nb_filter=nb_filter_conv5, nb_row=conv_filter_size[0], nb_col=conv_filter_size[1],
                      border_mode='same', subsample=conv_stride, init='glorot_normal')(conv4)
conv5 = MaxPooling2D(pool_size=pool_filter_size, strides=pool_stride)(conv5)
conv5 = Activation('elu')(conv5)
conv5 = Dropout(dropout_prob)(conv5)
#Layer6 (5x10x64)
conv6 = Convolution2D(nb_filter=nb_filter_conv6, nb_row=conv_filter_size[0], nb_col=conv_filter_size[1],
                      border_mode='same', subsample=conv_stride, init='glorot_normal')(conv5)
#conv6 = MaxPooling2D(pool_size=pool_filter_size, strides=pool_stride)(conv6) #no maxpooling
conv6 = Activation('elu')(conv6)
conv6 = Dropout(dropout_prob)(conv6)
#flatten before going to fully connected (5*10*64=3200 features)
conv6_flatten = Flatten()(conv6)
#Layer7 (100x3200)
fc7 = Dense(output_dim=nb_fc7, activation='elu')(conv6_flatten)
fc7 = Dropout(dropout_prob)(fc7)
#Layer8 (50x100)
fc8 = Dense(output_dim=nb_fc8, activation='elu')(fc7)
fc8 = Dropout(dropout_prob)(fc8)
#Layer9 (10x50)
fc9 = Dense(output_dim=nb_fc8, activation='elu')(fc8)
fc9 = Dropout(dropout_prob)(fc9)
#Output layer
prediction = Dense(output_dim=nb_prediction)(fc9) #no activation or dropout on the output layer
# and no softmax as its a regression problem and not classification

model = Model(input=inputs, output=prediction)
print(model.summary())

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

################ Non-generator version#########################
#model.load_weights('model.h5') #, by_name=True) #load previously save weights. don't do by_name=True for identical model. It doesn't laod the weights as the initial loss is still high doing so.

history = model.fit(features_train, steering_angle_train, batch_size=batch_size, nb_epoch=nb_epochs, verbose=1, 
                    validation_data=(features_val, steering_angle_val), shuffle=False)

final_val_acc = history.history['val_acc'][-1]
print('Final epochs validation accuracy: ', final_val_acc) #final epoch's validation accuracy

score = model.evaluate(features_val, steering_angle_val, verbose=0) #evaluate gives the loss and accuracy (if using accuracy metric in model.compile)
print(score)
print(model.metrics_names)

pred = model.predict(features_val[0:1])[0] #to keep it 4D tensor for a single example
print('prediciton:', pred) 
##############################################################


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

################ Non-generator version#########################
pred = model.predict(features_val[0:1])[0]
print('reloaded models prediciton:', pred) #to keep it 4D tensor for a single example
##############################################################

tf = time.time()
print('Time to train: {0:.2}seconds'.format(tf-to))