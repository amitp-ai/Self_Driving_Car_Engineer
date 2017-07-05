Project 3: Behavioral Cloning

The architecture used is from the Nvidia paper. Different networks based on the VGGNet as well as the architecture mentioned on comma.ai’s github account were tried, but none performed as well as the one used by Nvidia. The model uses 6 convolutional layers followed by 3 fully connected layers. At the input, there's also a special layer for data normalization. Moreover, I have included a layer at the input to learn the most optimal color space for the images. Given the depth of the architecture, the activation layer used is elu so as to avoid the "dead-neuron" problem associated with ReLUs as well as minimize the gradient saturation problem associated with sigmoid neurons.

Below is the summary of the final architecture used:

Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
input_1 (InputLayer)             (None, 80, 160, 3)    0                                            
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 80, 160, 3)    0           input_1[0][0]                    
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 80, 160, 3)    12          lambda_1[0][0]                   
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 80, 160, 3)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 80, 160, 24)   672         activation_1[0][0]               
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 40, 80, 24)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 40, 80, 24)    0           maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 40, 80, 24)    0           activation_2[0][0]               
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 40, 80, 36)    7812        dropout_1[0][0]                  
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 20, 40, 36)    0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 20, 40, 36)    0           maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 20, 40, 36)    0           activation_3[0][0]               
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 20, 40, 48)    15600       dropout_2[0][0]                  
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 10, 20, 48)    0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 10, 20, 48)    0           maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 10, 20, 48)    0           activation_4[0][0]               
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 10, 20, 64)    27712       dropout_3[0][0]                  
____________________________________________________________________________________________________
maxpooling2d_4 (MaxPooling2D)    (None, 5, 10, 64)     0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 5, 10, 64)     0           maxpooling2d_4[0][0]             
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 5, 10, 64)     0           activation_5[0][0]               
____________________________________________________________________________________________________
convolution2d_6 (Convolution2D)  (None, 5, 10, 64)     36928       dropout_4[0][0]                  
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 5, 10, 64)     0           convolution2d_6[0][0]            
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 5, 10, 64)     0           activation_6[0][0]               
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 3200)          0           dropout_5[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           320100      flatten_1[0][0]                  
____________________________________________________________________________________________________
dropout_6 (Dropout)              (None, 100)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dropout_6[0][0]                  
____________________________________________________________________________________________________
dropout_7 (Dropout)              (None, 50)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            2550        dropout_7[0][0]                  
____________________________________________________________________________________________________
dropout_8 (Dropout)              (None, 50)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             51          dropout_8[0][0]                  
====================================================================================================
Total params: 416487
____________________________________________________________________________________________________

One of the most important things I learned in this project is using python generators. The actual neural network implementation process was similar to the previous project, but using Keras instead of TensorFlow. Having said that though, this project really made me appreciate the empirical nature of deep learning. How theory sometimes doesn’t hold with the experiments. For instance, given we’re controlling a continuous variable (steering angle), I chose to use mean-squared error as the loss function. However, it really didn’t help much in predicting how well the car drives when tested. Similarly, while I generated validation data, the loss on the validation data didn’t help much in predicting which epoch performed the best on the actual track.

In terms of how the model was trained, I heavily relied on data augmentation instead of taking multiple-sets of data from different places where the car goes off track. I believe data augmentation allows the network to generalize better as it can learn about situations not easy to generate just by taking more data. E.g. in different brightness conditions etc. So for data augmentation, I randomly shifted the images left/right (and adjusted the steering angle accordingly). I also randomly shifted the image up/down to emulate driving up/down on a road. Moreover, I randomly changed the image brightness to emulate driving in different road conditions and I suspect this helped drive better on the challenge track as it is darker than the first track. I also used the data from multiple cameras and adjusted the steering angle accordingly.

In terms of actual training, I went through alot of different iterations. The final methodology I chose was to test over 16 epochs and save the model/weights for each and test each one. Usually the best epoch was between 7 and 13. Thereafter, loading the weights from the best epoch, I fine-tuned the model by significantly reducing the learning rate (lowered it to 4e-5 from 1e-3) using Adam optimizer. To make the model generalize better, I used dropout. I tried playing with L2 regularization, but it didn’t help.

I also modified drive.py in the following manner:
1. Crop the image (top 25% i.e. horizon) and bottom 10% (to match how the model was trained)
2. Resize the image to 80x160 pixels
3. Then feed the image to my neural network
4. To help pass the challenge track, I did two things:
	a. I amplified any steering angle predictions greater than 0.2 by a factor of 1.5X
	b. Since the challenge track a many steep roads to climb, and given the throttle is constant, if the speed drops below 6MPH, I increase the throttle

After completing all this, I was successfully able to drive on both tracks.



