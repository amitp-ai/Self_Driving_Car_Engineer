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
from sklearn.model_selection import train_test_split

def shuffle_driving_log():
    
    # Read the csv file containing the steering angle
    #driving_log = pd.read_csv('training_data/driving_log.csv', header=None)
    #driving_log = pd.read_csv('training_data/Another_Student_Data/driving_log.csv', header=None)
    driving_log = pd.read_csv('training_data/Udacity_Data/driving_log.csv') #, header=None) #remove header as udacity data does have it
    print(driving_log.ix[10][0])
    #print(driving_log.shape[0])

    #driving_log.ix = driving_log.ix.reindex(np.random.permutation(driving_log.shape[0])) #doesn't work
    #print(driving_log.ix[10][0]) #this doesn't work
    
    driving_log_temp = driving_log.copy()
    shuffled_array = np.random.permutation(driving_log.shape[0])

    for idx,val in enumerate(shuffled_array):
        driving_log.ix[idx] = driving_log_temp.ix[val]
 
    print(driving_log.ix[10][0])
    #print(driving_log.shape[0])

    del driving_log_temp #delete it to save memory
    driving_log.to_csv('training_data/Udacity_Data/shuffled_driving_log.csv', sep=',', header=False, index=False) #comma separated

def augment_brigtness(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) #in HSV space it is easier to change the brightness
    random_bright = 0.25+np.random.uniform() #offset by 0.25 instead of 0.5 since the original training is done in day time, random_bright is skewed with higher probability to making the image darker than brighter
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1
    
def augment_translate(image):
    rows,cols,clr = image.shape
    #y_tr = 0
    #x_tr = 0
    y_tr = (np.random.uniform()-0.5)*rows*0.1 #translate up/down to emulate the effect of driving up/down on a road. max translate is 10% of the image width
    x_tr = (np.random.uniform()-0.5)*cols*0.1 #translate left/right to emulate the effect of driving on corners. max translate is 10% of the image width
    M_translate = np.float32([[1,0,x_tr],[0,1,y_tr]])
    image1 = cv2.warpAffine(image, M_translate, (cols,rows))
    #for 10% shift in image in horizontal direction, change steering angle by 0.2. No change due to vertical shift in the image
    #negative so as to bring the car back to center
    steer_delta = -1*x_tr/(cols*0.1)*0.2 
    return image1, steer_delta

def augment_flip(image):
    rows,cols,clr = image.shape
    x_tr = cols 
    y_tr = 0 #don't affect the vertical direction
    M_flip = np.float32([[-1,0,x_tr],[0,1,y_tr]]) #-(x-cols/2)-cols/2 = cols-x where x_tr=cols
    image1 = cv2.warpAffine(image, M_flip, (cols,rows)) #can also use cv2.flip()
    return image1
    
def generate_data_generator(batch_size=128):
    
    # Read the csv file containing the steering angle
    #driving_log = pd.read_csv('training_data/driving_log.csv', header=None)
    #driving_log = pd.read_csv('training_data/Another_Student_Data/shuffled_driving_log.csv', header=None)
    driving_log = pd.read_csv('training_data/Udacity_Data/shuffled_driving_log.csv', header=None)
    images_prepend_path = 'training_data/Udacity_Data/'

    
    # Do driving_log.ix[rowID][colID] to access information
    # Column 1 is center image
    # Column 2 is left image
    # Column 3 is right image
    # Steering angle
    # Throttle
    # Brake
    # Speed
   
    total_size = driving_log.shape[0]
    
    assert (batch_size <= total_size), 'Batch size is too large!'
    num_images = batch_size
    
    image_file_name = images_prepend_path + driving_log.ix[0][0]
    image_center = mpimg.imread(image_file_name)
    crop_dim = (int(image_center.shape[0]*0.25),int(image_center.shape[0]*0.9))
            
    # original image shape is [160,320,3]
    new_image_dim = (160,80) #(320,160) #(160,80) #needs to be a tuple. List doesn't work. Reduce each dimension by 2 (4X reduction in area/pixels)
    images_shape = [num_images,new_image_dim[1],new_image_dim[0],3]
    
    full_camera_data = {}
    full_images_shape = images_shape
    full_num_images = num_images*3*4 #three cameras worth of data and (original, brightness adjust, translate adjust, and flip) for each
    full_images_shape[0] = full_num_images
    full_camera_data['features'] = np.zeros(full_images_shape)
    full_camera_data['steering_angle'] = np.zeros(full_num_images)
        
    i = 0
    while True:
        if (i + num_images) >= total_size:
            i = 0


        for j in range(num_images):
            #Read center camera image
            image_file_name = images_prepend_path + driving_log.ix[i][0]
            image_center = mpimg.imread(image_file_name) #.astype(np.uint8)
            image_center = image_center[crop_dim[0]:crop_dim[1]][:][:] #crop the horizon and car bumper out
            image_center = cv2.resize(image_center, new_image_dim, interpolation = cv2.INTER_AREA)
            center_steer_angle = driving_log.ix[i][3]
            
            #Read left camera image
            image_file_name = images_prepend_path + driving_log.ix[i][1][1:]
            image_left = mpimg.imread(image_file_name) #.astype(np.uint8)
            image_left = image_left[crop_dim[0]:crop_dim[1]][:][:] #crop the horizon and car bumper out
            image_left = cv2.resize(image_left, new_image_dim, interpolation = cv2.INTER_AREA)
            left_steer_angle = center_steer_angle + 0.25 #left camera
            
            #Read right camera image
            image_file_name = images_prepend_path + driving_log.ix[i][2][1:]       
            image_right = mpimg.imread(image_file_name) #.astype(np.uint8)
            image_right = image_right[crop_dim[0]:crop_dim[1]][:][:] #crop the horizon and car bumper out
            image_right = cv2.resize(image_right, new_image_dim, interpolation = cv2.INTER_AREA)
            right_steer_angle = center_steer_angle - 0.25 #right camera
                    
            #Center camera image
            full_camera_data['features'][0*num_images+j] = image_center
            full_camera_data['steering_angle'][0*num_images+j] = center_steer_angle
            #augment brightness
            image_bright = augment_brigtness(image_center)
            full_camera_data['features'][1*num_images+j] = image_bright
            full_camera_data['steering_angle'][1*num_images+j] = center_steer_angle
            #augment translate
            image_translate,steer_translate = augment_translate(image_center)
            full_camera_data['features'][2*num_images+j] = image_translate
            full_camera_data['steering_angle'][2*num_images+j] = center_steer_angle + steer_translate
            #augment flip
            image_flip = augment_flip(image_center)
            full_camera_data['features'][3*num_images+j] = image_flip
            full_camera_data['steering_angle'][3*num_images+j] = -1*center_steer_angle
            
            #left camera image
            full_camera_data['features'][4*num_images + j] = image_left
            full_camera_data['steering_angle'][4*num_images + j] = left_steer_angle
            #augment brightness
            image_bright = augment_brigtness(image_left)
            full_camera_data['features'][5*num_images+j] = image_bright
            full_camera_data['steering_angle'][5*num_images+j] = left_steer_angle
            #augment translate
            image_translate,steer_translate = augment_translate(image_left)
            full_camera_data['features'][6*num_images+j] = image_translate
            full_camera_data['steering_angle'][6*num_images+j] = left_steer_angle + steer_translate
            #augment flip
            image_flip = augment_flip(image_left)
            full_camera_data['features'][7*num_images+j] = image_flip
            full_camera_data['steering_angle'][7*num_images+j] = -1*left_steer_angle           
        
            #Right camera image
            full_camera_data['features'][8*num_images + j] = image_right
            full_camera_data['steering_angle'][8*num_images + j] = right_steer_angle
            #augment brightness
            image_bright = augment_brigtness(image_right)
            full_camera_data['features'][9*num_images+j] = image_bright
            full_camera_data['steering_angle'][9*num_images+j] = right_steer_angle
            #augment translate
            image_translate,steer_translate = augment_translate(image_right)
            full_camera_data['features'][10*num_images+j] = image_translate
            full_camera_data['steering_angle'][10*num_images+j] = right_steer_angle + steer_translate
            #augment flip
            image_flip = augment_flip(image_right)
            full_camera_data['features'][11*num_images+j] = image_flip
            full_camera_data['steering_angle'][11*num_images+j] = -1*right_steer_angle
            
            i += 1
        #To randomly shuffle the training set
        features_train, _, steer_train, _ = train_test_split(full_camera_data['features'], 
                                                             full_camera_data['steering_angle'], test_size=1e-6, random_state=0)
        #print(12*num_images,'\n')
        steer_train *= 1.0 #make the steering angles slightly more aggressive?
        yield (features_train, steer_train) #one less data point than expected due to validation data generated above (minimum=1)


#This is only executed if this file is explicitly run
if __name__ == '__main__':
    shuffle_driving_log()

######################################################################
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
#5. add regularization, and control learning rate