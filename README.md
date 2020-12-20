# Self-Driving Car Engineer Nanodegree
This repository contains all the projects I completed as part of Udacity's self-driving car engineer nanodegree. 

The Self-Driving Car Engineer Nanodegree is an online certification intended to prepare students to become self-driving car engineers. The program was developed by Udacity in partnership with Mercedes-Benz, Nvidia, Uber ATG, amongst others.

As detailed below, the program covered a wide range of topics including traditional computer vision, deep learning, sensor fusion, localization, path-planning, control, etc. 

## Program Outline:

## Term 1: Computer Vision and Deep Learning (Fall 2016)
**Traditional Computer Vision (Python)**
* Project 1 - Finding Lane Lines (Intro to Computer Vision): Introductory project which used basic computer vision techniques like canny edge and hough transforms to detect lane lines
* Project 4 - Advanced Lane Lines (Computer Vision): Use of image thresholding, warping and fitting lanes lines to develop a more robust method of detecting lane lines on a road
* Project 5 - Vehicle Detection (Computer Vision): Use of HOG and SVM to detect vehicles on a road

**Deep Learning (Python/Tensorflow)**
* Project 2 - Traffic Sign Classifier (Deep Learning): Train a convolution neural network capable of detecting road side traffic signs.
* Project 3 - Behavioral Cloning (Deep Learning): Train a car to drive in a 3D simulator using a deep neural network (input to the network is an RGB image and the output is the corresponding steering angle).


## Term 2: Sensor Fusion, Localization, and Control (Spring 2017)
**Sensor Fusion (C++)**
* Project 1: Combine lidar and radar data to track objects in the environment using Extended Kalman filter (EKF) 
* Project 2: Combine lidar and radar data to track objects in the environment using Unscented Kalman filter (UKF) 


**Localization (C++)**
* Project 3: Localize the EGO vehicle relative to the world map using a particle filter.

**Control (C++)**
* Project 4: Use PID control to steer the car as well as adjust brake/acceleration inorder to follow a reference trajectory.
* Project 5: Use Model Predictive Control (MPC) to develop a more advanced controller that can handle hardware response delays as well as real-world actuator constraints. 


## Term 3: Path Planning, Semantic Segmentation, and System Integration (Summer 2017)
**Path Planning (C++)**
* Project 1: Generate a smooth trajectory (as well as proper speed) to navigate the vehicle on a highway while avoiding obstacles and other vehicles. Additionally, find a sequence of steps in a maze to navigate cities, parking lots, etc.

**Semantic Segmentation (Python/Tensorflow)**
* Project 2: Using Fully-Convolutional Network (FCN) based semantic segmentation architecture, classified each pixel in the image into road, car, or everything else category.

**Capstone Project (C++ and Python/Tensorflow)**
* Project 3: A system integration team project to run on a real self-driving car. My task was to develop a traffic light detection module using the Single-Shot Detection (SSD) network. The network would output the location of the traffic light (bounding box) as well as the traffic light state (red, green, yellow, or not working).
