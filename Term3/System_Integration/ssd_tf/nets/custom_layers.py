# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 17:22:42 2017

@author: APatel7
"""

# Copyright 2015 Paul Balanca. All Rights Reserved.
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
"""Implement some custom layers, not provided by TensorFlow.

Trying to follow as much as possible the style/standards used in
tf.contrib.layers
"""
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import nn
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope

#NOTE: THE INPUT DATA IS ASSUMED TO BE IN NHWC FORMAT (NUM_IMAGE, HEIGHT, WIDTH, CHANNELS)

def abs_smooth(x):
    """Smoothed absolute function. Useful to compute an L1 smooth error.

    Define as:
        x^2 / 2         if abs(x) < 1
        abs(x) - 0.5    if abs(x) > 1
    We use here a differentiable definition using min(x) and abs(x). Clearly
    not optimal, but good enough for our purpose!
    """
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx) #this will create a tensorflow operation and not a python operation as the variables are tensors
    return r


 #to decorate the below function so that it can be used within an arg_scope.
@add_arg_scope
def l2_normalization(
        inputs,
        scaling=False,
        scale_initializer=init_ops.ones_initializer(),
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        scope=None):
    """Implement L2 normalization on every feature (i.e. spatial normalization).

    Should be extended in some near future to other dimensions, providing a more
    flexible normalization framework.

    Args:
      inputs: a 4-D tensor with dimensions [batch_size, height, width, channels].
      scaling: whether or not to add a post scaling operation along the dimensions
        which have been normalized.
      scale_initializer: An initializer for the weights.
      reuse: whether or not the layer and its variables should be reused. To be
        able to reuse the layer scope must be given.
      variables_collections: optional list of collections for all the variables or
        a dictionary containing a different list of collection per variable.
      outputs_collections: collection to add the outputs.
      trainable: If `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
      scope: Optional scope for `variable_scope`.
    Returns:
      A `Tensor` representing the output of the operation.
    """

    with variable_scope.variable_scope(
            scope, 'L2Normalization', [inputs], reuse=reuse) as sc:
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims #gives the dimension of the tensor without running a session
        dtype = inputs.dtype.base_dtype

        # norm_dim = tf.range(1, inputs_rank-1)
        norm_dim = tf.range(inputs_rank-1, inputs_rank) #i.e. normalize along the channel dimension
        params_shape = inputs_shape[-1:]

        # Normalize along spatial dimensions.
        outputs = nn.l2_normalize(inputs, norm_dim, epsilon=1e-12)
        # Additional scaling (the scaling factor is learned)
        if scaling:
            scale_collections = utils.get_variable_collections(
                variables_collections, 'scale')
            scale = variables.model_variable('gamma',
                                             shape=params_shape,
                                             dtype=dtype,
                                             initializer=scale_initializer,
                                             collections=scale_collections,
                                             trainable=trainable)
            outputs = tf.multiply(outputs, scale) #for conv4, scale is learned to be 20 (as per the SSD paper)

        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope, outputs)

 #to decorate the below function so that it can be used within an arg_scope.
@add_arg_scope
def pad2d(inputs,
          pad=(0, 0),
          mode='CONSTANT',
          trainable=True,
          scope=None):
    """2D Padding layer, adding a symmetric padding to H and W dimensions.

    Aims to mimic padding in Caffe and MXNet, helping the port of models to
    TensorFlow. Tries to follow the naming convention of `tf.contrib.layers`.

    Args:
      inputs: 4D input Tensor;
      pad: 2-Tuple with padding values for H and W dimensions;
      mode: Padding mode. C.f. `tf.pad`
    """
    with tf.name_scope(scope, 'pad2d', [inputs]):
        # Padding shape.
        # Pad with zeros in the width and height dimensions (HW). No padding for the N and C dimensions.
        paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]

        net = tf.pad(inputs, paddings, mode=mode)
        return net