# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 10:54:14 2016

@author: amit_p
"""
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.image as mpimg #read in images
import matplotlib.pyplot as plt
import cv2
import math


def generate_data():
    
    # Read the csv file containing the steering angle
    #driving_log = pd.read_csv('training_data/driving_log.csv', header=None)
    driving_log = pd.read_csv('training_data/Another_Student_Data/driving_log.csv', header=None)
    images_prepend_path = 'training_data/Another_Student_Data/IMG/'
    
    # Do driving_log.ix[rowID][colID] to access information
    # Column 1 is center image
    # Column 2 is left image
    # Column 3 is right image
    # Steering angle
    # Throttle
    # Brake
    # Speed
   
    #Downsamping effectively reduces the sampling rate of the simulator.
    #Because the data in the driving_log.csv file is added chronologically, 
    #this has same effect as if the simulator was saving images at 10Hz instead of 50Hz default (using a 5x downsample)
    #We don't need such a high sampling rate for training the car as the car is not moving so fast.
    down_sample_factor = 1 #4
    total_size = driving_log.shape[0]
    num_images = math.ceil(driving_log.shape[0]/down_sample_factor)
    
    # original image shape is [160,320,3]
    new_image_dim = (160,80) #(320,160) #(160,80) #needs to be a tuple. List doesn't work. Reduce each dimension by 2 (4X reduction in area/pixels)
    images_shape = [num_images,new_image_dim[1],new_image_dim[0],3]

#    center_camera = {}
#    center_camera['features'] = np.zeros(images_shape)
#    center_camera['steering_angle'] = np.zeros(num_images)
#    
#    left_camera = {}
#    left_camera['features'] = np.zeros(images_shape)
#    left_camera['steering_angle'] = np.zeros(num_images)
#    
#    right_camera = {}
#    right_camera['features'] = np.zeros(images_shape)
#    right_camera['steering_angle'] = np.zeros(num_images)
#    
#    camera_data = {}
#    camera_data['center_cam'] = center_camera
#    camera_data['left_cam'] = left_camera
#    camera_data['right_cam'] = right_camera
    
    full_camera_data = {}
    full_images_shape = images_shape
    full_images_shape[0] = num_images*3 #three cameras worth of data
    full_camera_data['features'] = np.zeros(full_images_shape)
    full_camera_data['steering_angle'] = np.zeros(num_images*3) #three cameras worth of data
    
    for i,j in zip(range(0,total_size,down_sample_factor), range(num_images)): #reduce the data by downsample factor
        #Read input images
        #image_file_name = driving_log.ix[i][0][65:] #as the center camera name does not start with space
        image_file_name = images_prepend_path + driving_log.ix[i][0][9:]
        image = mpimg.imread(image_file_name) #.astype(np.uint8)
        image = cv2.resize(image, new_image_dim, interpolation = cv2.INTER_AREA)
        #center_camera['features'][j] = image
        #plt.imshow(center_camer['features'][j])
        #center_camera['steering_angle'][j] = driving_log.ix[i][3]
        full_camera_data['features'][j] = image
        full_camera_data['steering_angle'][j] = driving_log.ix[i][3]
    
        #Read input images
        #image_file_name = driving_log.ix[i][1][66:] #as the left camera name starts with space
        image_file_name = images_prepend_path + driving_log.ix[i][1][10:]        
        image = mpimg.imread(image_file_name) #.astype(np.uint8)
        image = cv2.resize(image, new_image_dim, interpolation = cv2.INTER_AREA)
        #left_camera['features'][j] = image
        #plt.imshow(center_camer['features'][j])
        #left_camera['steering_angle'][j] = driving_log.ix[i][3]-0.1 #left camera
        full_camera_data['features'][num_images + j] = image
        full_camera_data['steering_angle'][num_images + j] = driving_log.ix[i][3]-0.1 #left camera
    
        #Read input images
        #image_file_name = driving_log.ix[i][2][66:] #as the right camera name starts with space
        image_file_name = images_prepend_path + driving_log.ix[i][2][10:]        
        image = mpimg.imread(image_file_name) #.astype(np.uint8)
        image = cv2.resize(image, new_image_dim, interpolation = cv2.INTER_AREA)
        #right_camera['features'][j] = image
        #plt.imshow(center_camer['features'][j])
        #right_camera['steering_angle'][j] = driving_log.ix[i][3]+0.1 #right camera
        full_camera_data['features'][num_images*2 + j] = image
        full_camera_data['steering_angle'][num_images*2 + j] = driving_log.ix[i][3]+0.1 #right camera


    return full_camera_data

#The save file gets too big. So just process the data directly
#data_file = 'camera_training_data.p'
##save data
#with open(data_file, 'wb') as f: #write in binary format
#    pickle.dump(center_camera, f) #gives memory error on laptop if use camera_data dicitonary
#    
##load data
#with open(data_file, 'rb') as f: #read in binary format
#    camera_data_loaded = pickle.load(f)


#Here's the code for using the generators approach if need be
##Use a generator for as not to run into memory limitations
#def batch_generator(<information about csv file .. bla blah etc. >, batch_size=32):
#    num_rows = <figure out total number of rows in csv>
#    train_images = np.zeros((batch_size, img_rows, img_cols, 3))
#    train_steering = np.zeros(batch_size)
#    ctr = None
#    while 1:        
#        for j in range(batch_size):
#            if ctr is None or ctr >= num_rows:
#                ctr = 0     # Initialize counter or reset counter if over bounds
#            train_images[j], train_steering[j] = <load data from row number "ctr" in the CSV file> 
#            ctr += 1
#        yield train_images, train_steering

#another example        
#def generate_arrays_from_file(path):
#    while 1:
#    f = open(path)
#    for line in f:
#        # create Numpy arrays of input data
#        # and labels, from each line in the file
#        x, y = process_line(line)
#        yield (x, y)
#    f.close()
#
#model.fit_generator(generate_arrays_from_file('/my_file.txt'),
#        samples_per_epoch=10000, nb_epoch=10)

#to do
#1. use generators
#2. shift images up/down and left/right
#3. 
# vivek yadav
#its all hapazard i think. Biut after all thnis, came up with these steps, 
#1- Augmentation: changing brighness, adding left and right images, translating images left and right and adding corresponding 
#angles, flipping images.
#2- Subsample training set so you are more likely to keep data that has higher turn angle. 
#I did this in the initial stages, and kept increasing the probability of keeping data after each step.
#4. maybe also use throttle or speed or even steering angle to predict the next steering angle
