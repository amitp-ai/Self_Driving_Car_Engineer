#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 20:19:59 2018

@author: amit
"""

import matplotlib.pyplot as plt
#import matplotlib.image as mpimg #don't use this, use scipy.misc
import scipy.misc
import numpy as np
from glob import glob
import os.path
import time
import random
import yaml


#pre-process gt_labels to only identify vehicles and roads 
def preprocess_labels(label_image):
    ##debug
    #labels_new = np.copy(label_image)
    #plt.imshow(labels_new[:,:,0])
    #plt.show()
    #labels_new[labels_new==10] = 0
    #print(9.0/labels_new[100,100,0]) #gives 255 as the normalizer using mpimg and 1 as the normalizer using scipy.misc
    #print(10.0/labels_new[590,100,:]) #gives 255 as the normalizer using mpimg and 1 as the normalizer using scipy.misc
    #print(7.0/labels_new[400,300,:]) #gives 255 as the normalizer using mpimg and 1 as the normalizer using scipy.misc
    ##end debug

    #note: np.nonzero() returns the elements/indices of the numpy array
    #label_image *= 255 #gt labels are normalized between 0 and 1 (do only if using mpimg and not scipy.misc)
    label_image = np.around(label_image,decimals=0)
    labels_new = np.copy(label_image)
    # Identify lane marking pixels (label is 6)
    lane_marking_pixels = (label_image[:,:,0] == 6).nonzero()
    # Set lane marking pixels to road (label is 7)
    #print(len(lane_marking_pixels))
    labels_new[lane_marking_pixels] = 7 #applies 7 to all the channels

    # Identify all vehicle pixels
    vehicle_pixels = (label_image[:,:,0] == 10).nonzero()
    # Isolate vehicle pixels associated with the hood (y-position > 496)
    hood_indices = (vehicle_pixels[0] >= 496).nonzero()[0]
    hood_pixels = (vehicle_pixels[0][hood_indices], \
                   vehicle_pixels[1][hood_indices])
    # Set hood pixel labels to 0
    labels_new[hood_pixels] = 0 #applies 0 to all the channels
    
    # set all other labels to 0
    for id in range(0,13):
        if (id != 7) and (id != 10):
            other_pixels = (labels_new[:,:,0] == id).nonzero()
            labels_new[other_pixels] = 0 #applies 0 to all the channels
    
    #For proper one-hot-key encoding as there are only 3 class types
    #set id=7 (roads/lanes) to 1 and set id=10 (vehicles) to 2
    seg_pixels = (labels_new[:,:,0] == 7).nonzero()
    #print(len(seg_pixels))
    labels_new[seg_pixels] = 1 #applies 1 to all the channels as seg_pixels is length 2
    seg_pixels = (labels_new[:,:,0] == 10).nonzero()
    labels_new[seg_pixels] = 2 #applies 2 to all the channels as seg_pixels is length 2
    
    #Really not needed to make the last two channels zero everywhere as the last two channels won't be used
    labels_new[:,:,1] = 0
    labels_new[:,:,2] = 0
    
    # Return the preprocessed label image 
    return labels_new



if __name__ == '__main__':
    generate_Preprocessed_gt = False
    generate_Train_Test_Data = True
    print("generate_Preprocessed_gt is: ", generate_Preprocessed_gt)
    print("generate_Train_Test_data is: ", generate_Train_Test_Data)
    if (generate_Preprocessed_gt == True):
        #Before running make sure there is an empty folder 
        #named 'PreProcessedCameraSeg' in both './data/Addnl_Data' and './data/Orig_Data'
        
        orig_image_gt_data_folder = './data/Orig_Data/CameraSeg' #original data
        #orig_image_gt_data_folder = './data/Addnl_Data/CameraSeg' #additional data from carla  simulator
        image_gt_paths = glob(os.path.join(orig_image_gt_data_folder, '*.png'))
        for image_gt_path in image_gt_paths:
            image_gt = scipy.misc.imread(image_gt_path)
            #For Debug
            #print(image_gt_path)
            #print(image_gt.shape)
            #plt.imshow(image_gt[:,:,2])
            #plt.show()

            preprocessed_image_gt = preprocess_labels(image_gt)
            preprocessed_image_gt_path = image_gt_path.replace('CameraSeg','PreProcessedCameraSeg')
            scipy.misc.imsave(preprocessed_image_gt_path, preprocessed_image_gt)
            #For Debug
            #print((scipy.misc.imread(preprocessed_image_gt_path)).shape)
            #image_gt = preprocessed_image_gt
            #plt.imshow(image_gt[:,:,0])
            #plt.show()

    if(generate_Train_Test_Data == True):
        orig_image_gt_data_folder = './data/Orig_Data/CameraRGB' #original data
        image_gt_paths = glob(os.path.join(orig_image_gt_data_folder, '*.png'))

        #orig_image_gt_data_folder = './data/Addnl_Data/CameraRGB' #additional data from carla  simulator
        #image_gt_paths = image_gt_paths + glob(os.path.join(orig_image_gt_data_folder, '*.png')) #combine the two lists of file names
        random.shuffle(image_gt_paths)
        print(len(image_gt_paths))

        num_test_data = 50
        num_val_data = 50

        test_data_yaml = './data/test.yaml'
        val_data_yaml = './data/val.yaml'
        train_data_yaml = './data/train.yaml'

        test_data = image_gt_paths[:num_test_data]
        val_data = image_gt_paths[num_test_data:(num_test_data+num_val_data)]
        train_data = image_gt_paths[(num_test_data+num_val_data):]

        #write the data in yaml file
        with open(test_data_yaml, 'w') as yaml_file:
            yaml.dump(test_data, yaml_file, default_flow_style=False)
        with open(val_data_yaml, 'w') as yaml_file:
            yaml.dump(val_data, yaml_file, default_flow_style=False)
        with open(train_data_yaml, 'w') as yaml_file:
            yaml.dump(train_data, yaml_file, default_flow_style=False)

        #To read the data:
        #training_data_list = yaml.load(open(train_data_yaml, 'rb').read())        

