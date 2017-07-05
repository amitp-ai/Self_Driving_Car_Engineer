# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 12:09:27 2016

@author: amit_p
"""
import numpy as np
import pandas as pd
import matplotlib.image as mpimg #read in images
import matplotlib.pyplot as plt
from training_data_generation_generator_randomized_2 import *
import cv2

#a = generate_data_generator()
#b = a.__next__()
##
##shuffle_driving_log()

#images_prepend_path = 'training_data/Udacity_Data/'
#
#driving_log = pd.read_csv('training_data/Udacity_Data/shuffled_driving_log.csv', header=None)
#image_file_name = images_prepend_path + driving_log.ix[100][0]
#image = mpimg.imread(image_file_name) #.astype(np.uint8)
#plt.imshow(image)
#plt.show()
#
#image1 = augment_brigtness(image)
#plt.imshow(image1)
#plt.show()
#
#image1,_ = augment_translate(image)
#plt.imshow(image1)
#plt.show()
#
#image1 = augment_flip(image)
#plt.imshow(image1)
#plt.show()

#a=generate_data_generator(batch_size=8)
#for i in range(5):
#    print(a.__next__()[1].shape)
sel = (1,0,1)
cam = (1,0,2)
for i in range(1):
    driving_log = pd.read_csv('training_data/Udacity_Data/driving_log.csv', header=None)
    images_prepend_path = 'training_data/Udacity_Data/'
    image_file_name = images_prepend_path + driving_log.ix[10][cam[i]][sel[i]:]
    image = mpimg.imread(image_file_name) #.astype(np.uint8)
    plt.imshow(image)
    plt.show()
    st_angle = float(driving_log.ix[10][3])
    print(st_angle)
    image,st_angle = augment_translate(image,st_angle)
    plt.imshow(image)
    plt.show()
    print(st_angle)
    
#image_file_name = images_prepend_path + driving_log.ix[10][0]
#image = mpimg.imread(image_file_name) #.astype(np.uint8)
#plt.imshow(image)
#plt.show()
#image_file_name = images_prepend_path + driving_log.ix[10][2][1:]
#image = mpimg.imread(image_file_name) #.astype(np.uint8)
#plt.imshow(image)
#plt.show()

#crop_dim = (int(image_center.shape[0]*0.25),int(image_center.shape[0]*0.9))
#plt.imshow(image_center[crop_dim[0]:crop_dim[1]])
#plt.show()

#image_center = mpimg.imread(file_name)
#plt.imshow(image_center)
#plt.show()
#image_center2 = cv2.resize(image_center, (160,80), interpolation = cv2.INTER_AREA)
#plt.imshow(image_center2)
#plt.show()
#image_center = image_center[crop_dim[0]:crop_dim[1]] #[:][:]
#plt.imshow(image_center)
#plt.show()
#image_center = cv2.resize(image_center, (160,80), interpolation = cv2.INTER_AREA)
#plt.imshow(image_center)
#plt.show()
#image_centerf = augment_flip(image_center)
#plt.imshow(image_centerf)
#plt.show()
#image_centerf = cv2.flip(image_center,1)
#plt.imshow(image_centerf)
#plt.show()

#from PIL import Image
#image = Image.open(image_file_name)
#Image._show(image)
#imagec = image.crop((0,int(image.size[1]*0.25),image.size[0],int(image.size[1]*0.9))) #added by Amit
#Image._show(imagec)


