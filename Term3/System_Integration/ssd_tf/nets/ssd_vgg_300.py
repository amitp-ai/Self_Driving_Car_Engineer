# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 17:18:24 2017

@author: APatel7
"""

# This is based off of: https://github.com/balancap/SSD-Tensorflow/blob/master/nets/ssd_vgg_300.py

# Copyright 2016 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Definition of 300 VGG-based SSD network.

This model was initially introduced in:
SSD: Single Shot MultiBox Detector
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
Cheng-Yang Fu, Alexander C. Berg
https://arxiv.org/abs/1512.02325

Two variants of the model are defined: the 300x300 and 512x512 models, the
latter obtaining a slightly better accuracy on Pascal VOC.

Usage:
    with slim.arg_scope(ssd_vgg.ssd_vgg()):
        outputs, end_points = ssd_vgg.ssd_vgg(inputs)

This network port of the original Caffe model. The padding in TF and Caffe
is slightly different, and can lead to severe accuracy drop if not taken care
in a correct way!

In Caffe, the output size of convolution and pooling layers are computing as
following: h_o = (h_i + 2 * pad_h - kernel_h) / stride_h + 1

Nevertheless, there is a subtle difference between both for stride > 1. In
the case of convolution:
    top_size = floor((bottom_size + 2*pad - kernel_size) / stride) + 1
whereas for pooling:
    top_size = ceil((bottom_size + 2*pad - kernel_size) / stride) + 1
Hence implicitely allowing some additional padding even if pad = 0. This
behaviour explains why pooling with stride and kernel of size 2 are behaving
the same way in TensorFlow and Caffe.

Nevertheless, this is not the case anymore for other kernel sizes, hence
motivating the use of special padding layer for controlling these side-effects.

@@ssd_vgg_300
"""
import math
from collections import namedtuple

import numpy as np
import tensorflow as tf

from ssd_tf.tf_extended import bboxes as tfe_bboxes
from ssd_tf.nets import custom_layers
from ssd_tf.nets import ssd_common

#slim documentation: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
slim = tf.contrib.slim

#NOTE: THE INPUT DATA IS ASSUMED TO BE IN NHWC FORMAT (NUM_IMAGE, HEIGHT, WIDTH, CHANNELS)


# =========================================================================== #
# SSD class definition.
# =========================================================================== #
SSDParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])


class SSDNet(object):
    """Implementation of the SSD VGG-based 300 network.

    The default features layers with 300x300 image input are:
      conv4 ==> 38 x 38
      conv7 ==> 19 x 19
      conv8 ==> 10 x 10
      conv9 ==> 5 x 5
      conv10 ==> 3 x 3
      conv11 ==> 1 x 1
    The default image size used to train this network is 300x300.
    """
    default_params = SSDParams(
        img_shape=(300, 300), #use vgg300
        num_classes=6, #0,1,2,3,4,5 (background, red, yellow, green, off, occluded/unknown)
        no_annotation_label=6,
        feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11'],
        feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)], #output feature size
        anchor_size_bounds=[0.15, 0.90],
        # anchor_size_bounds=[0.20, 0.90],
        anchor_sizes=[(21., 45.), #This is used after being normalized to image_shape in the anchor_box generation function
                      (45., 99.), #Also, each number in anchor_sizes is scale i.e. sqrt(w*h)
                      (99., 153.),
                      (153., 207.),
                      (207., 261.),
                      (261., 315.)],
#        anchor_sizes=[(11., 25.), #This is added by Amit but doesn't help. Also, this is used after being normalized to image_shape in the anchor_box generation function
#                      (25., 49.), #Also, each number in anchor_sizes is scale i.e. sqrt(w*h)
#                      (49., 75.),
#                      (75., 103.),
#                      (103., 131.),
#                      (131., 161.)], 
       # anchor_sizes=[(30., 60.),
        #               (60., 111.),
        #               (111., 162.),
        #               (162., 213.),
        #               (213., 264.),
        #               (264., 315.)],
        anchor_ratios=[[2, .5], #TODO: need to verify this
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5],
                       [2, .5]],
        anchor_steps=[8, 16, 32, 64, 100, 300], #This is same as effective stride. But not sure of blocks 10 and 11
        anchor_offset=0.5, #TODO: need to verify this
        normalizations= [20, -1, -1, -1, -1, -1], #As per the paper, only normalize the output of conv_4 and not the feature maps
        prior_scaling=[0.1, 0.1, 0.2, 0.2] #TODO: need to verify this
        )

    def __init__(self, params=None):
        """Init the SSD net with some parameters. Use the default ones
        if none provided.
        """
        if isinstance(params, SSDParams):
            self.params = params
        else:
            self.params = SSDNet.default_params

    # ======================================================================= #
    def net(self, inputs,
            is_training=True,
            update_feat_shapes=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_300_vgg'):
        """SSD network definition.
        """
        r = ssd_net(inputs,
                    num_classes=self.params.num_classes,
                    feat_layers=self.params.feat_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    anchor_ratios=self.params.anchor_ratios,
                    normalizations=self.params.normalizations,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    prediction_fn=prediction_fn,
                    reuse=reuse,
                    scope=scope)
        # Update feature shapes (try at least!)
        if update_feat_shapes:
            shapes = ssd_feat_shapes_from_net(r[0], self.params.feat_shapes)
            self.params = self.params._replace(feat_shapes=shapes)
        return r

    
    def arg_scope(self, weight_decay=0.0005, data_format='NHWC'):
        """Network arg_scope.
        """
        return ssd_arg_scope(weight_decay, data_format=data_format)
    
    # ======================================================================= #
    def update_feature_shapes(self, predictions): #TODO: Not sure what this is for
        """Update feature shapes from predictions collection (Tensor or Numpy
        array).
        """
        shapes = ssd_feat_shapes_from_net(predictions, self.params.feat_shapes)
        self.params = self.params._replace(feat_shapes=shapes)

    def anchors(self, img_shape, dtype=np.float32):
        """Compute the default anchor boxes, given an image shape.
        """
        return ssd_anchors_all_layers(img_shape,
                                      self.params.feat_shapes,
                                      self.params.anchor_sizes,
                                      self.params.anchor_ratios,
                                      self.params.anchor_steps,
                                      self.params.anchor_offset,
                                      dtype)

    def bboxes_encode(self, labels, bboxes, anchors,
                      scope=None):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_encode(
            labels, bboxes, anchors,
            self.params.num_classes,
            self.params.no_annotation_label,
            ignore_threshold=0.5,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def bboxes_decode(self, feat_localizations, anchors,
                      scope='ssd_bboxes_decode'):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_decode(
            feat_localizations, anchors,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def detected_bboxes(self, predictions, localisations,
                        select_threshold=None, nms_threshold=0.5,
                        clipping_bbox=None, top_k=400, keep_top_k=200):
        """Get the detected bounding boxes from the SSD network output.
        """
        # Select top_k bboxes from predictions, and clip
        rscores, rbboxes = \
            ssd_common.tf_ssd_bboxes_select(predictions, localisations,
                                            select_threshold=select_threshold,
                                            num_classes=self.params.num_classes)
        rscores, rbboxes = \
            tfe_bboxes.bboxes_sort(rscores, rbboxes, top_k=top_k)
        # Apply NMS algorithm.
        rscores, rbboxes = \
            tfe_bboxes.bboxes_nms_batch(rscores, rbboxes,
                                 nms_threshold=nms_threshold,
                                 keep_top_k=keep_top_k)
        if clipping_bbox is not None:
            rbboxes = tfe_bboxes.bboxes_clip(clipping_bbox, rbboxes)
        return rscores, rbboxes

    def losses(self, logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope='ssd_losses'):
        """Define the SSD network losses.
        """
        return ssd_losses(logits, localisations,
                          gclasses, glocalisations, gscores,
                          match_threshold=match_threshold,
                          negative_ratio=negative_ratio,
                          alpha=alpha,
                          label_smoothing=label_smoothing,
                          scope=scope)


# =========================================================================== #
# SSD tools...
# =========================================================================== #
def ssd_size_bounds_to_values(size_bounds,
                              n_feat_layers,
                              img_shape=(300, 300)):
    """Compute the reference sizes of the anchor boxes from relative bounds.
    The absolute values are measured in pixels, based on the network
    default size (300 pixels).

    This function follows the computation performed in the original
    implementation of SSD in Caffe.

    Return:
      list of list containing the absolute sizes at each scale. For each scale,
      the ratios only apply to the first value.
    """
    assert img_shape[0] == img_shape[1]

    img_size = img_shape[0]
    min_ratio = int(size_bounds[0] * 100)
    max_ratio = int(size_bounds[1] * 100)
    step = int(math.floor((max_ratio - min_ratio) / (n_feat_layers - 2)))
    # Start with the following smallest sizes.
    sizes = [[img_size * size_bounds[0] / 2, img_size * size_bounds[0]]]
    for ratio in range(min_ratio, max_ratio + 1, step):
        sizes.append((img_size * ratio / 100.,
                      img_size * (ratio + step) / 100.))
    return sizes


def ssd_feat_shapes_from_net(predictions, default_shapes=None):
    """Try to obtain the feature shapes from the prediction layers. The latter
    can be either a Tensor or Numpy ndarray.

    Return:
      list of feature shapes. Default values if predictions shape not fully
      determined.
    """
    feat_shapes = []
    for l in predictions:
        # Get the shape, from either a np array or a tensor.
        if isinstance(l, np.ndarray):
            shape = l.shape
        else:
            shape = l.get_shape().as_list()
        shape = shape[1:4]
        # Problem: undetermined shape...
        if None in shape:
            return default_shapes
        else:
            feat_shapes.append(shape)
    return feat_shapes

#This function has been verified. It is used to generate anchor boxes.
def ssd_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
    """Computer SSD default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      img_shape: Image shape, used for computing height, width relative to the image;
      feat_shape: Feature shape, used for computing relative position grids;
      size: Anchor sizes for the given layer (Absolute reference sizes)
      ratios: Anchor Ratios for the given layer/feature map;
      step: Anchor Step size for the given layer (mostly equivalent to the effective stride for each feature map)
      offset: anchor offset i.e. Grid offset.

    Return:
      y, x, h, w: Relative (center coordinate) x and y grids, and height and width.
    """
    # Compute the position grid: simple way.
    # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    # y = (y.astype(dtype) + offset) / feat_shape[0]
    # x = (x.astype(dtype) + offset) / feat_shape[1]
    # Weird SSD-Caffe computation using steps values...
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]

    #center coordinates are offset'ed by half-grid i.e. 0.5
    #then it is scaled by the effective stride to the input image
    #then it is normalized by dividing by the input_image_shape
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1) #adds a dimension at the end
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Note: scale = sqrt(area) = sqrt(w*h)
    # Note: aspect_ratio = h/w
    # Note: w = scale*sqrt(aspect_ratio)
    # Note: h = scale/sqrt(aspect_ratio)
    # Note: sizes[0] = scale1 and sizes[1]=scale2
    # Note: sqrt(sizes[0]*sizes[1]) is the geometric mean of the two scales
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(sizes) + len(ratios)
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)
    # Add first anchor boxes with ratio=1.
    h[0] = sizes[0] / img_shape[0] #aspect ratio=1
    w[0] = sizes[0] / img_shape[1] #aspect_ratio=1
    di = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0] #aspect ratio=1
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1] #aspect_ratio=1
        di += 1
    for i, r in enumerate(ratios):
        h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
    
    #Note: the size of y/x is different than h/w
    #For each y/x (center coordinate), have all the h/w anchorboxes.
    #Number of h/w = number of anchor boxes per center coordinate
    #Number of x/y = number of center coordinates (which is determined by np.mgrid())
    return y, x, h, w

#This function has been verified. It is used to generate anchor boxes.
def ssd_anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = ssd_anchor_one_layer(img_shape, s,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors


# =========================================================================== #
# Functional definition of VGG-based SSD 300.
# =========================================================================== #
#This function has been verified
#It is called by ssd_multibox_layer() function
def tensor_shape(x, rank=3):
    """Returns the dimensions of a tensor.
    Args:
      image: A N-D Tensor of shape.
    Returns:
      A list of dimensions. Dimensions that are statically known are python
        integers,otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]

#This function has been verified
#It is called by ssd_net() function
def ssd_multibox_layer(inputs,
                       num_classes,
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False):
    """Construct a multibox layer, return a class and localization predictions.
    inputs: conv feature map (4d tensor) output from a specific layer in the vgg-16 network
    num classes: number of classes
    sizes: anchor sizes
    ratios: anchor ratios
    normalization: 
    returns the class predictions and localization prediction
    """
    net = inputs
    if normalization > 0:
        net = custom_layers.l2_normalization(net, scaling=True) #for conv4, scale is learned to be 20 (as per the SSD paper)
    # Number of anchors.
    num_anchors = len(sizes) + len(ratios)

    # Location.
    num_loc_pred = num_anchors * 4
    loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None,
                           scope='conv_loc')
    #loc_pred is: NHWxnum_loc_pred
    #So reshape it to: NHWxnum_anchorsx4 (reshape from 4D to 5D tensor)
    loc_pred = tf.reshape(loc_pred,
                          tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 4])
    # Class prediction.
    num_cls_pred = num_anchors * num_classes
    cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None,
                           scope='conv_cls')
    #cls_pred is: NHWxnum_cls_pred
    #So reshape it to: NHWxnum_anchorsxnum_classes (reshape from 4D to 5D tensor)
    cls_pred = tf.reshape(cls_pred,
                          tensor_shape(cls_pred, 4)[:-1]+[num_anchors, num_classes])
    return cls_pred, loc_pred

#This function has been verified
def ssd_net(inputs,
            num_classes=SSDNet.default_params.num_classes,
            feat_layers=SSDNet.default_params.feat_layers,
            anchor_sizes=SSDNet.default_params.anchor_sizes,
            anchor_ratios=SSDNet.default_params.anchor_ratios,
            normalizations=SSDNet.default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_300_vgg'):
    """SSD net definition.
    """
    # End_points collect relevant activations for external use.
    end_points = {}
    with tf.variable_scope(scope, 'ssd_300_vgg', [inputs], reuse=reuse):
        # Original VGG-16 blocks.
        #Block 1.
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1') #repeats conv2d 2x and default stride is 1 and padding is 'SAME'
        end_points['block1'] = net #block1: effective stride=1
        #print('1', net.get_shape().as_list())
        net = slim.max_pool2d(net, [2, 2], scope='pool1') #default stride is 2 and padding is 'VALID'
        #print('1M', net.get_shape().as_list())
        # Block 2.
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2') #repeats conv2d 2x and default stride is 1 and padding is 'SAME'
        end_points['block2'] = net #block2: effective stride=2
        #print('2', net.get_shape().as_list())
        #1. Explicitly pad here otherwise the later layers don't match with the SSD paper 
        #2. Also otherwise doesn't match feat_shapes defined in default parameters
        #3. Also otherwise it gives error in the final block11 layer
        net = custom_layers.pad2d(net, pad=(1, 1)) #Not present in the original TF github repo
        #print('2P', net.get_shape().as_list())
        net = slim.max_pool2d(net, [2, 2], scope='pool2') #default stride is 2 and padding is 'VALID'
        #print('2M', net.get_shape().as_list())
        # Block 3.
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3') #repeats conv2d 3x and default stride is 1 and padding is 'SAME'
        end_points['block3'] = net #block3: effective stride=4
        #print('3', net.get_shape().as_list())
        net = slim.max_pool2d(net, [2, 2], scope='pool3') #default stride is 2 and padding is 'VALID'
        #print('3M', net.get_shape().as_list())
        # Block 4.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4') #repeats conv2d 3x and default stride is 1 and padding is 'SAME'
        end_points['block4'] = net #block4: effective stride=8
        #print('4', net.get_shape().as_list())
        net = slim.max_pool2d(net, [2, 2], scope='pool4') #default stride is 2 and padding is 'VALID'
        # Block 5.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5') #repeats conv2d 3x and default stride is 1 and padding is 'SAME'
        end_points['block5'] = net #block5: effective stride=16
        # For the below max pool, in the original tf sdd github code the padding is 'VALID'
        # But then the sizes of blocks 6 and 7 don't match with the SSD paper
        # So have changed the padding to 'SAME'
        net = slim.max_pool2d(net, [3, 3], stride=1, padding='SAME', scope='pool5')
        # Additional SSD blocks.

        # Block 6: rate parameter is used for atrous convolution. see tf.contrib.layers.conv2d
        net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6') #default stride is 1 and padding is 'SAME'
        end_points['block6'] = net #block6: effective stride=16
        net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training) #rate=dropout rate
        # Block 7: 1x1 conv.
        net = slim.conv2d(net, 1024, [1, 1], scope='conv7') #default stride is 1 and padding is 'SAME'
        end_points['block7'] = net #block7: effective stride=16
        net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)

        # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
        end_point = 'block8'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        end_points[end_point] = net #block8: effective stride=32
        #print('8', net.get_shape().as_list())
        end_point = 'block9'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        end_points[end_point] = net #block9: effective stride=64
        #print('9', net.get_shape().as_list())
        end_point = 'block10'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        end_points[end_point] = net #block10: effective stride=64
        end_point = 'block11'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
            #print(net.get_shape().as_list())
        end_points[end_point] = net #block11: effective stride=64

        # Prediction and localizations layers.
        predictions = []
        logits = []
        localizations = []
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box'):
                p, l = ssd_multibox_layer(end_points[layer],
                                          num_classes,
                                          anchor_sizes[i],
                                          anchor_ratios[i],
                                          normalizations[i])
            predictions.append(prediction_fn(p)) #slim.softmax is same as tf.nn.softmax()
            logits.append(p)
            localizations.append(l)

        return predictions, localizations, logits, end_points
ssd_net.default_image_size = 300 #FIXME: I dont' know where this is used.


def ssd_arg_scope(weight_decay=0.0005, data_format='NHWC'):
    """Defines the VGG arg scope.

    Args:
      weight_decay: The l2 regularization coefficient.

    Returns:
      An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format) as sc:

            return sc

# =========================================================================== #
# SSD loss function.
# =========================================================================== #
def ssd_losses(logits, localizations,
               gclasses, glocalizations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope=None):
    """Loss functions for training the SSD 300 VGG network.

    This function defines the different loss components of the SSD, and
    adds them to the TF loss collection.

    Arguments:
      logits: (list of) predictions logits Tensors;
      localisations: (list of) localisations Tensors;
      gclasses: (list of) groundtruth labels Tensors;
      glocalisations: (list of) groundtruth localisations Tensors;
      gscores: (list of) groundtruth score Tensors;
      match_threshold: to compare the gscores (basically the threshold above which we accept the gscores)
      negative_ratio: ratio of positive to negative(hard negative mining) examples
    """
    with tf.name_scope(scope, 'ssd_losses'):
        lshape = tensor_shape(logits[0], 5)
        num_classes = lshape[-1]
        batch_size = lshape[0]

        # Flatten out all vectors!
        flogits = []
        flocalizations = []
        fgclasses = []
        fgscores = []
        fglocalizations = []
        for i in range(len(logits)):
            flogits.append(tf.reshape(logits[i], [-1, num_classes]))
            flocalizations.append(tf.reshape(localizations[i], [-1, 4]))
            fgclasses.append(tf.reshape(gclasses[i], [-1]))
            fgscores.append(tf.reshape(gscores[i], [-1])) #TODO: Find out how gscores is generated
            fglocalizations.append(tf.reshape(glocalizations[i], [-1, 4]))
        # And concat everything!
        logits = tf.concat(flogits, axis=0)
        gclasses = tf.concat(fgclasses, axis=0)
        gscores = tf.concat(fgscores, axis=0)
        localizations = tf.concat(flocalizations, axis=0)
        glocalizations = tf.concat(fglocalizations, axis=0)
        dtype = logits.dtype
        
        # Compute positive matching mask...
        pmask = gscores > match_threshold
        fpmask = tf.cast(pmask, dtype)
        n_positives = tf.reduce_sum(fpmask)

        # Hard negative mining...
        no_classes = tf.cast(pmask, tf.int32)
        predictions = slim.softmax(logits)
        #'tf.logical_not(pmask)' is same as 'gscores <= match_threshold(i.e. 0.5)'
        #essentially nmask is 1 where: '-0.5 < gscores < 0.5' and 0 otherwise
        #nmask is of same shape as gscores and is 1-dimensional
        nmask = tf.logical_and(tf.logical_not(pmask),
                               gscores > -0.5)
        fnmask = tf.cast(nmask, dtype) #fnmask is of same shape as gscores and is 1-dimensional
        
        #predictions[:,0] is of same shape as gscores and is 1-dimensional
        #nvalues has the same dimension as nmask. It takes elementwise value from predictions[:, 0] if nmask=1
        #And it takes elementwise value from '1.0-fnmask' if nmask=0
        nvalues = tf.where(nmask,
                           predictions[:, 0],
                           1. - fnmask)
        nvalues_flat = tf.reshape(nvalues, [-1])
        # Number of negative entries to select.
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
        n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + batch_size
        n_neg = tf.minimum(n_neg, max_neg_entries)
        #
        #given nvals is a vector (1D tensor), it will return a vector of length k containing the value and idx
        #of the top k elements in '-nvalues_flat'
        val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
        max_hard_pred = -val[-1]
        # Final negative mask.
        nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
        fnmask = tf.cast(nmask, dtype)

        batch_size_flt32 = tf.cast(batch_size, tf.float32)
        # Add cross-entropy loss.
        with tf.name_scope('cross_entropy_pos'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=gclasses)
            loss = tf.div(tf.reduce_sum(loss * fpmask), batch_size_flt32, name='value')
            tf.losses.add_loss(loss)

        with tf.name_scope('cross_entropy_neg'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=no_classes)
            loss = tf.div(tf.reduce_sum(loss * fnmask), batch_size_flt32, name='value')
            tf.losses.add_loss(loss)

        # Add localization loss: smooth L1, L2, ...
        with tf.name_scope('localization'):
            # Weights Tensor: positive mask + random negative.
            weights = tf.expand_dims(alpha * fpmask, axis=-1)
            loss = custom_layers.abs_smooth(localizations - glocalizations) #smooth L1 loss
            loss = tf.div(tf.reduce_sum(loss * weights), batch_size_flt32, name='value')
            tf.losses.add_loss(loss)
        
        #use tf.losses.get_total_loss() to get the above losses as well as any regularized losses
        