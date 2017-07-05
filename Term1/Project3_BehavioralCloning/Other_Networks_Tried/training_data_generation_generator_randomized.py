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

def augment_brigtness(image,st_angle):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) #in HSV space it is easier to change the brightness
    random_bright = 0.25+np.random.uniform() #offset by 0.25 instead of 0.5 since the original training is done in day time, random_bright is skewed with higher probability to making the image darker than brighter
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    st_angle = st_angle + 0
    return image1, st_angle
    
def augment_translate(image, st_angle):
    rows,cols,clr = image.shape
    #y_tr = 0
    #x_tr = 0
    y_tr = (np.random.uniform()-0.5)*rows*0.2 #translate up/down to emulate the effect of driving up/down on a road. max translate is 20% of the image height
    x_tr = (np.random.uniform()-0.5)*cols*0.3 #translate left/right to emulate the effect of having different camera mounts. max translate is 30% of the image width
    M_translate = np.float32([[1,0,x_tr],[0,1,y_tr]])
    image1 = cv2.warpAffine(image, M_translate, (cols,rows))
    #for 10% shift in image in horizontal direction, change steering angle by 0.2. No change due to vertical shift in the image
    #negative so as to bring the car back to center
    steer_delta = 1*x_tr/(cols*0.3)*0.4
    st_angle = st_angle + steer_delta
    return image1, st_angle

def augment_flip(image,st_angle):
    rows,cols,clr = image.shape
    x_tr = cols 
    y_tr = 0 #don't affect the vertical direction
    M_flip = np.float32([[-1,0,x_tr],[0,1,y_tr]]) #-(x-cols/2)-cols/2 = cols-x where x_tr=cols
    image1 = cv2.warpAffine(image, M_flip, (cols,rows)) #can also use cv2.flip()
    st_angle = st_angle*-1
    return image1, st_angle
    
def generate_data_generator(batch_size=16):
    
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
    full_num_images = num_images*2 #original and augmented image
    full_images_shape[0] = full_num_images
    full_camera_data['features'] = np.zeros(full_images_shape)
    full_camera_data['steering_angle'] = np.zeros(full_num_images)
        
    i = 0
    while True:
        if (i + num_images) >= total_size:
            i = 0

        for j in range(num_images):
            img_sel = np.random.randint(3)

            if img_sel == 0:
                #Read left camera image
                image_file_name = images_prepend_path + driving_log.ix[i][1][1:]
                image = mpimg.imread(image_file_name) #.astype(np.uint8)
                image = image[crop_dim[0]:crop_dim[1]][:][:] #crop the horizon and car bumper out
                image = cv2.resize(image, new_image_dim, interpolation = cv2.INTER_AREA)
                steer_angle = driving_log.ix[i][3] + 0.1 #left camera
                
            elif img_sel == 1:
                #Read center camera image
                image_file_name = images_prepend_path + driving_log.ix[i][0]
                image = mpimg.imread(image_file_name) #.astype(np.uint8)
                image = image[crop_dim[0]:crop_dim[1]][:][:] #crop the horizon and car bumper out
                image = cv2.resize(image, new_image_dim, interpolation = cv2.INTER_AREA)
                steer_angle = driving_log.ix[i][3]
            
            elif img_sel == 2:    
                #Read right camera image
                image_file_name = images_prepend_path + driving_log.ix[i][2][1:]       
                image = mpimg.imread(image_file_name) #.astype(np.uint8)
                image = image[crop_dim[0]:crop_dim[1]][:][:] #crop the horizon and car bumper out
                image = cv2.resize(image, new_image_dim, interpolation = cv2.INTER_AREA)
                steer_angle = driving_log.ix[i][3] - 0.1 #right camera
                    
            #Original image
            full_camera_data['features'][0*num_images+j] = image
            full_camera_data['steering_angle'][0*num_images+j] = steer_angle
            
            #augmented image
            #augment brightness
            image, steer_angle = augment_brigtness(image, steer_angle)
            image, steer_angle = augment_translate(image, steer_angle)
            img_flip = np.random.randint(2)
            if img_flip == 1: #else don't flip
                image, steer_angle = augment_flip(image, steer_angle)
                
            full_camera_data['features'][1*num_images+j] = image
            full_camera_data['steering_angle'][1*num_images+j] = steer_angle
            
            i += 1
        #To randomly shuffle the training set
        features_train, _, steer_train, _ = train_test_split(full_camera_data['features'], 
                                                             full_camera_data['steering_angle'], test_size=1e-6, random_state=0)
        #print(12*num_images,'\n')
        ###steer_train *= 1.3 #1.3 makes it really bad and it still fails at the same corner
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