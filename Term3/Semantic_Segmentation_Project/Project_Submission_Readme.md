## Rubric Requirements
Note: All the tests in the main.py file pass successfully.

1. The function 'load_vgg' is implemented to basically load the pretrained vgg network and its weights. However the entire network will be trained end-to-end after adding the below mentioned modifications.

2. The function 'layers' is implemented to convert the output layer of the vgg (layer 7) into a fully convolutional layer with a 1x1 convolution with 2 filters (as there are two classes). Similarly, the output of fully connected layers 3 and 4 are converted into fully convolutional layers using 1x1 convolutions and 2 filters (again due to 2 classes). These fully connected layers are converted in to fully convolutional layers to maintain spatial information as well as classify each individual pixel. Note that the 1x1 convolution is performed to the output of a given layer without adding any non-linearity in between. For example, the layer 7 outpt is fed into a 1x1 convolution layer without adding any non-linearity in between. Similar is the case for the outputs of layers 3 and 4.

To improve spatial resolution of the output segmentation predictions, the higher resolution outputs of layers 3 and 4 are combined (using skip layers) with the output of layer 7.

In particular, the output of layer 7 is first upsampled by 2x using transposed convolutions. It is then additively combined with the output of layer 4. This combination is then upsampled again by 2x and combined with the fully convolutionaly output from layer 3. This combination is then upsampled by 8x so that the dimensions of the final output is same as the input image's dimension. Moreover, due to the skip connections, as mentioned previously, the output layer has high spatial resolution. Just as with the convolution layers, the transposed convolution layers' weights are learned.

Note, just as with the 1x1 convolutions, no non-linearity is added across the transposed convolution layer. It is just performing learnable (linear) matrix operations to shape the output into a desired shape.

3. The function 'optimize' is properly implemented using cross entropy softmax loss function and minimized using the Adam optimizer -- which has built in laearning rate decay and other enhancements to the standard SGD algorithm. For better generalization, regularization is performed. However, regularization is implemneted in the 'layers' function mentioned above in the conv2d and transposed_convolution functions. All the additional layers are regularized using a lambda of 1e-3.

4. The function 'train_nn' is implemented to end-to-end train the modified vggnet for semantic segmentation. 

5. Training is performed over 25 epochs (took about 30minutes on a Google Cloud K80 GPU). The softmax entropy loss consistently decreases to about 0.05. The final IOU is about 95% for training and validation data -- which is take off of the training dataset. A batch size of 32 is used as it is not too large to make the GPU run out of memory but not too small to affect training accuracy.

As shown in the runs directory, the trained networks has performed pretty well on the test (i.e. inference) data.

6. The model is saved to allow for further training or inference without having to re-train from the beginning.



