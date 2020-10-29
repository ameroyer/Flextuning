import os
import sys
from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

"""Net architectures"""


############################################### MNIST Classifier
def mnist_net(images,
              num_outputs=10,
              is_training=True, 
              reuse=False, 
              is_chief=True, 
              verbose=True, 
              **kwargs):
    """ Small MNIST classification network 
        c.f. https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/mnist/mnist_deep.py
    
    Args:
        images: (batch, width, height, 1) Tensor of input images in [0, 1]
        is_training: whether we are in training mode or not   
        reuse: whether to reuse the variable scopes
        is_chief: whether the model is run by the chief worker
        verbose: verbosity level
        kwargs: remaining keyword arguments (unused here)
        
    Returns:
        A (batch, 10) Tensor of unscaled logits
    """
    del is_chief
    del kwargs
    keep_prob = 0.5 if is_training else 1.0
    
    # Input : (28, 28, 1)
    with tf.control_dependencies([tf.assert_greater_equal(images, 0.)]):
        with tf.control_dependencies([tf.assert_less_equal(images, 1.)]):
            net = images
    if verbose: print('   \033[34min:\033[0m', net.get_shape()) 
    
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, normalizer_fn=None, padding='SAME', stride=1):
        with slim.arg_scope([slim.max_pool2d], kernel_size=[2, 2], stride=2, padding='SAME'):
            # Conv 1: (?, 14, 14, 32)
            with tf.variable_scope('layer_1', reuse=reuse):
                net = slim.conv2d(net, 32, [5, 5], scope='conv_1')
                net = slim.max_pool2d(net)
                if verbose: print('   \033[34mconv1:\033[0m', net.get_shape())
            # Conv 2: (?, 7, 7, 64)
            with tf.variable_scope('layer_2', reuse=reuse):
                net = slim.conv2d(net, 64, [5, 5], scope='conv_2')
                net = slim.max_pool2d(net)
                if verbose: print('   \033[34mconv2:\033[0m', net.get_shape())
            # Fc 1: (?, 1024)
            with tf.variable_scope('layer_3', reuse=reuse):
                net = tf.layers.flatten(net)
                net = slim.fully_connected(net, 1024, scope='fc_1')
                net = tf.nn.relu(net)
                with tf.name_scope('dropout'):
                    net = slim.layers.dropout(net, keep_prob=keep_prob)       
                if verbose: print('   \033[34mfc:\033[0m', net.get_shape())   
            # Fc 1: (?, 10)  
            with tf.variable_scope('layer_4', reuse=reuse):
                net = slim.fully_connected(net, num_outputs, activation_fn=None, scope='fc_2')
                if verbose: print('   \033[34mout:\033[0m', net.get_shape())
            return net
        
        
############################################### CIFAR Classifier
def cifar_net(images, is_training=True, reuse=False, is_chief=True, verbose=True, **kwargs):
    """ CIFAR classification network (- class deer)
        c.f. https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/mnist/mnist_deep.py
    
    Args:
        images: (batch, width, height, 1) Tensor of input images in [0, 1]
        is_training: whether we are in training mode or not   
        reuse: whether to reuse the variable scopes
        is_chief: whether the model is run by the chief worker
        verbose: verbosity level
        kwargs: remaining keyword arguments (unused here)
        
    Returns:
        A (batch, 10) Tensor of unscaled logits
    """
    del is_chief
    del kwargs
    keep_prob = 0.5 if is_training else 1.0
    
    # Input : (32, 32, ?)
    with tf.control_dependencies([tf.assert_greater_equal(images, 0.)]):
        with tf.control_dependencies([tf.assert_less_equal(images, 1.)]):
            net = images
    if verbose: print('   \033[34min:\033[0m', net.get_shape()) 
    
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, normalizer_fn=None, padding='SAME', stride=1):
        with slim.arg_scope([slim.max_pool2d], kernel_size=[2, 2], stride=2, padding='SAME'):
            # Conv 1:  (?, 32, 32, 32)
            with tf.variable_scope('layer_1', reuse=reuse):
                net = slim.conv2d(net, 32, [3, 3], scope='conv_1')
                if verbose: print('   \033[34mconv1:\033[0m', net.get_shape())
            # Conv 2: (?, 16, 16, 64)
            with tf.variable_scope('layer_2', reuse=reuse):
                net = slim.conv2d(net, 64, [3, 3], scope='conv_2')
                net = slim.max_pool2d(net)
                if verbose: print('   \033[34mconv2:\033[0m', net.get_shape())
            # Conv 3: (?, 8, 8, 128)
            with tf.variable_scope('layer_3', reuse=reuse):
                net = slim.conv2d(net, 128, [3, 3], scope='conv_3')
                net = slim.max_pool2d(net)
                if verbose: print('   \033[34mconv3:\033[0m', net.get_shape())
            # Conv 4: (?, 4, 4, 256)
            with tf.variable_scope('layer_4', reuse=reuse):
                net = slim.conv2d(net, 256, [3, 3], scope='conv_4')
                net = slim.max_pool2d(net)
                if verbose: print('   \033[34mconv4:\033[0m', net.get_shape())
            # Conv 5: (?, 2, 2, 512)
            with tf.variable_scope('layer_5', reuse=reuse):
                net = slim.conv2d(net, 512, [3, 3], scope='conv_5')
                net = slim.max_pool2d(net)
                if verbose: print('   \033[34mconv5:\033[0m', net.get_shape())
            # Fc 1: (?, 1024)
            with tf.variable_scope('layer_6', reuse=reuse):
                net = tf.layers.flatten(net)
                net = slim.fully_connected(net, 1024, scope='fc_1')
                net = tf.nn.relu(net)
                with tf.name_scope('dropout'):
                    net = slim.layers.dropout(net, keep_prob=keep_prob)       
                if verbose: print('   \033[34mfc:\033[0m', net.get_shape())   
            # Fc 1: (?, 10)  
            with tf.variable_scope('layer_7', reuse=reuse):
                net = slim.fully_connected(net, 9, activation_fn=None, scope='fc_2')
                if verbose: print('   \033[34mout:\033[0m', net.get_shape())
            return net
        
       
############################################### Inception based classifiers 
def inception_net(images,
                  num_outputs=1000,
                  is_training=True,
                  reuse=False,
                  is_chief=True,
                  verbose=False,
                  **kwargs):
    """Inceptionv2 from tensornets (https://github.com/taehoonlee/tensornets)
    
    Args:
        images: (batch, width, height, 3) Tensor of input images in [0, 1]
        num_outputs: Number of outputs
        is_training: whether we are in training mode or not   
        reuse: whether to reuse the variable scopes
        is_chief: whether the model is run by the chief worker
        verbose: verbosity level
        kwargs: remaining keyword arguments (unused here)
        
    Returns:
        A (batch, num_outputs) Tensor of unscaled logits
    """
    del is_chief
    del kwargs
    # Load tensornets
    sys.path.append('/nfs/scistore12/chlgrp/aroyer/Libs/tensornets')
    import tensornets
    # Preprocess images
    with tf.control_dependencies([tf.assert_greater_equal(images, 0.)]):
        with tf.control_dependencies([tf.assert_less_equal(images, 1.)]):
            images = (images - 0.5) * 2.
    if verbose: print('   \033[34min:\033[0m', images.get_shape()) 
    # Inception v2
    logits = tensornets.Inception2(images, 
                                   is_training=is_training,
                                   classes=num_outputs,
                                   scope='InceptionV2',
                                   reuse=reuse)
    if verbose: print('   \033[34mlogits:\033[0m', logits.get_shape()) 
    return logits

"""ILSVRC Classifier: 1000-way output"""
ilsvrc_net = partial(inception_net, num_outputs=1000)    

"""PACS Classifier: Inception net where the last fully connected-layer has 7 outputs"""
pacs_net = partial(inception_net, num_outputs=7)

"""PACS (from ImageNet) classifier: A standard ILSVRC classifier where the 1000 output logits 
are mapped to a subset of 7 classes matching the classes in PACS"""
pacs_to_imagenet_mapping = np.array([167, 385, 354, 402, 339, 449, 610])

def pacs_from_imagenet_net(images, verbose=False, **kwargs):
    global pacs_to_imagenet_mapping
    logits = inception_net(images, verbose=False, **kwargs)
    logits = tf.gather(logits, pacs_to_imagenet_mapping, axis=-1)
    if verbose: print('   \033[34mlogits:\033[0m', logits.get_shape()) 
    return logits



############################################### PREPROCESSERS
#### Only use for generative experiments (Section 5.4 in the paper)

## Generally take outputs in [-1., 1.]        
def pix2pix(images,
            encoder_blocks,
            kernel_size=4,
            num_outputs=3,
            variance_init=0.02,
            normalizer='instance_norm',
            with_backwards_connections=True,
            use_conv_transpose=True,
            upconv_pad=2,
            bottleneck_dims=-1,
            is_training=True,
            reuse=False,
            is_chief=True,
            verbose=False,
            **kwargs):
    """Pix2Pix inspired preprocessing network
    
    Args:
        images: a 4D tensor of input images in [-1, 1.]
        encoder_blocks: A list of integers representing the number of filters for each conv block.
        kernel_size: Convolution kernel size
        num_outputs: Number of outputs channels in the last layer
        normalizer: One of `instance_norm`, `batch_norm` or None
        with backwards_connections: Whether to use U-Net skip connections. Defaults to True
        use_conv_transpose: whether to use transpose convolutions; otherwise use upscale (nearest) + convolution
        upconv_pad: convolution padding after upscaling if use_conv_transpose is False
        bottleneck_dims: if > 0, use a fully connected bottleneck with the given numer of dimensions
        is_training: whether we are in training mode or not 
        reuse: Whether to reuse the model variables
        is_chief: whether the model is run by the chief worker
        verbose: verbosity level
        kwargs: remaining keyword arguments (unused here)
        
    Returns:
        A 4D Tensor of images in [0., 1.]
    """
    del is_chief
    del kwargs
    
    ## Config
    weights_initializer = tf.random_normal_initializer(0, variance_init)
    if normalizer == 'instance_norm':
        normalizer_fn = slim.instance_norm
        normalizer_params  = {'center': True, 'scale': True, 'epsilon': 1e-5}
    elif normalizer == 'batch_norm':
        normalizer_fn = slim.batch_norm
        normalizer_params = {'is_training': is_training, 'decay': 0.9, 'epsilon': 1e-5}
    elif normalizer is None:
        normalizer_fn = None
        normalizer_params = None
    else:
        raise ValueError('Unknown normalizer option', normalizer)
        
    ## input is in [-1, 1]   
    with tf.control_dependencies([tf.assert_greater_equal(images, -1.)]):
        with tf.control_dependencies([tf.assert_less_equal(images, 1.)]):
            net = images
            if verbose: print('   \033[34min:\033[0m', net.get_shape()) 
                
    ## Encoder
    activation_fn = tf.nn.leaky_relu
    with tf.contrib.framework.arg_scope([slim.conv2d],
                                        kernel_size=[kernel_size, kernel_size], 
                                        padding='SAME',
                                        stride=2,
                                        activation_fn=None,
                                        normalizer_fn=normalizer_fn,
                                        normalizer_params=normalizer_params,
                                        weights_initializer=weights_initializer):
        encoder_activations = []
        with tf.variable_scope('encoder', reuse=reuse):
            for block_id, block_num_filters in enumerate(encoder_blocks):
                scope = 'conv_%d' % (block_id + 1)
                if block_id > 0:
                    net = activation_fn(net)                                      # activation
                net = slim.conv2d(net, block_num_filters, scope=scope)  # conv2d + normalization
                encoder_activations.append(net)                         # save for skip-connection
                if verbose: print('   \033[34m%s:\033[0m' % scope, net.get_shape())
        if verbose: print('   \033[34mencoder-out:\033[0m', net.get_shape())   
            
    ## (optional) bottleneck
    if bottleneck_dims > 0:
        # net: (, w, h, c)
        if verbose: print('   \033[34mbottleneck-in:\033[0m', net.get_shape())   
        _, width, height, num_channels = net.get_shape().as_list()
        # net: (, 1, 1, dims)
        net = tf.layers.flatten(net)
        net = slim.fully_connected(net, bottleneck_dims)
        net = tf.nn.relu(net)
        net = tf.reshape(net, (-1, 1, 1, bottleneck_dims))
        # net: (, w, h, c)
        net = slim.conv2d_transpose(net, num_channels, [width, height], padding='VALID')
        if verbose: print('   \033[34mbottleneck-out:\033[0m', net.get_shape())
                        
    ## Decoder
    activation_fn = tf.nn.relu
    encoder_activations = encoder_activations[::-1]
    decoder_blocks = encoder_blocks[::-1]
    decoder_blocks[-1] = num_outputs
    assert(len(encoder_activations) == len(decoder_blocks))
    with tf.contrib.framework.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                        kernel_size=[kernel_size, kernel_size], 
                                        padding='SAME',
                                        activation_fn=None,
                                        weights_initializer=weights_initializer):
        with tf.contrib.framework.arg_scope([slim.conv2d], stride=1):
            with tf.contrib.framework.arg_scope([slim.conv2d_transpose], stride=2):
                with tf.variable_scope('decoder', reuse=reuse):
                    for block_id, block_num_filters in enumerate(decoder_blocks):
                        scope = 'deconv_%d' % (block_id + 1)
                        is_last_block = (block_id == len(decoder_blocks) - 1)
                        norm_fn = None if is_last_block else normalizer_fn
                        norm_params = None if is_last_block else normalizer_params
                        # concatenate skip connection
                        if with_backwards_connections and block_id > 0:
                            net = tf.concat([net, encoder_activations[block_id]], axis=3)
                        # activate
                        net = activation_fn(net)       
                        # conv-transpose
                        if use_conv_transpose:
                            net = slim.conv2d_transpose(net, block_num_filters, scope=scope, 
                                                        normalizer_fn=norm_fn, normalizer_params=norm_params) 
                        # upscale + conv
                        else:
                            target_size = net.get_shape()[1] * 2 - upconv_pad * 2
                            net = tf.image.resize_nearest_neighbor(net, (target_size, target_size))
                            net = tf.pad(net, ((0, 0), (upconv_pad, upconv_pad), (upconv_pad, upconv_pad), (0, 0)))
                            net = slim.conv2d(net, block_num_filters, scope=scope, 
                                              normalizer_fn=norm_fn, normalizer_params=norm_params) 
                        if verbose: print('   \033[34mdeconv%d\033[0m:' % (block_id + 1), net.get_shape())
            
                with tf.variable_scope('output', reuse=reuse):
                    net = tf.nn.tanh(net)
                    if verbose: print('   \033[34mout:\033[0m', net.get_shape())    
                    return net
          
    
def spatial_transformer(images,
                        encoder_blocks=[128],
                        is_training=True,
                        reuse=False,
                        is_chief=True,
                        verbose=False,
                        **kwargs):
    """A simple Spatial Transformer Network constrained to TSR style transformation.
    
    Args:
        images: a 4D tensor of input images in [0., 1.]
        encoder_blocks: A list of integers indicating the number of each channel in the encoder. The last layer of the encoder 
            is fully-connected, while the rests are convolutional blocks with leaky ReLU and batch norm.
        is_training: whether we are in training mode or not 
        reuse: Whether to reuse the model variables
        is_chief: whether the model is run by the chief worker
        verbose: verbosity level
        kwargs: remaining keyword arguments (unused here)
        
    Returns:
        A 4D Tensor of images in [0., 1.]
    """
    del is_chief
    del kwargs
    
    # Use STN from https://github.com/kevinzakka/spatial-transformer-network
    sys.path.append('spatial-transformer-network')
    from stn import spatial_transformer_network as transformer
    with tf.control_dependencies([tf.assert_greater_equal(images, 0.)]):
        with tf.control_dependencies([tf.assert_less_equal(images, 1.)]):
            net = images 
            in_dims = images.get_shape().as_list()[1:]
    
    with tf.variable_scope('localization_network', reuse=reuse):
        ## Encoder
        with tf.contrib.framework.arg_scope([slim.conv2d],
                                            kernel_size=[3, 3], 
                                            padding='SAME',
                                            stride=2,
                                            activation_fn=tf.nn.leaky_relu,
                                            normalizer_fn=slim.batch_norm,
                                            normalizer_params={'is_training': is_training, 'decay': 0.9, 'epsilon': 1e-5},
                                            weights_initializer=tf.random_normal_initializer(0, 0.02) ):
            for block_id, block_num_filters in enumerate(encoder_blocks[:-1]):
                scope = 'conv_%d' % (block_id + 1)                                   
                net = slim.conv2d(net, block_num_filters, scope=scope) 
                if verbose: print('   \033[34m%s:\033[0m' % scope, net.get_shape())
                    
        ## STN
        # fc 1
        net = tf.layers.flatten(net)
        net = slim.fully_connected(net, encoder_blocks[-1], activation_fn=tf.nn.tanh,
                                   weights_initializer=tf.zeros_initializer(),
                                   biases_initializer=tf.truncated_normal_initializer(stddev=0.01))
        if verbose: print('   \033[34mfc1:\033[0m', net.get_shape())
        # rotation angle (init: 0)
        theta = slim.fully_connected(net, 1, activation_fn=tf.nn.tanh,
                                    weights_initializer=tf.zeros_initializer(),
                                    biases_initializer=tf.truncated_normal_initializer(stddev=0.01)) * np.pi 
        rotate_matrix = tf.concat([tf.cos(theta),
                                   - tf.sin(theta),
                                   tf.zeros(tf.shape(theta)),
                                   tf.sin(theta), 
                                   tf.cos(theta),
                                   tf.zeros(tf.shape(theta)),
                                   tf.zeros(tf.shape(theta)),
                                   tf.zeros(tf.shape(theta)), 
                                   tf.ones(tf.shape(theta))],
                                  axis=-1)
        rotate_matrix = tf.reshape(rotate_matrix, (-1, 3, 3))
        # translation and scale (init: identity)
        translate_matrix = slim.fully_connected(net, 4, activation_fn=None,
                                                weights_initializer=tf.zeros_initializer(),
                                                biases_initializer=tf.constant_initializer([1., 1., 0., 0.])) #sx, sy, tx, tx
        translate_matrix = tf.split(translate_matrix, 4, axis=-1)
        translate_matrix = tf.concat([translate_matrix[0],
                                      tf.zeros(tf.shape(theta)),
                                      translate_matrix[2],
                                      tf.zeros(tf.shape(theta)),
                                      translate_matrix[1],
                                      translate_matrix[3],
                                      tf.zeros(tf.shape(theta)),
                                      tf.zeros(tf.shape(theta)),
                                      tf.ones(tf.shape(theta))], axis=1)
        translate_matrix = tf.reshape(translate_matrix, (-1, 3, 3))
        # final transformation
        transform_matrix = tf.matmul(rotate_matrix, translate_matrix)
        transform_matrix = tf.layers.flatten(transform_matrix)
        transform_matrix = transform_matrix[:, :6]
        images = transformer(images, transform_matrix, out_dims=in_dims)
        images = tf.clip_by_value(images, 0., 1.)
        return images


def conv_1x1_net(images,
                 bottleneck_dims=0,
                 is_training=True,
                 is_chief=True, 
                 reuse=False,
                 verbose=True, 
                 **kwargs):
    """Small network with one or two 1x1 convolutions with stride 1.
    
    Args:
        images: (batch, width, height, 1) Tensor of input images in [-1., 1.]
        bottleneck_dims: Number of channels in the first convolutional layer. if 0, only learns one convolution.
        is_training: whether we are in training mode or not 
        reuse: Whether to reuse the model variables
        is_chief: whether the model is run by the chief worker
        verbose: verbosity level
        kwargs: remaining keyword arguments (unused here)
        
    Returns:
        A 4D Tensor of images in [0., 1.]
    """
    del is_chief
    del kwargs
    
    with tf.control_dependencies([tf.assert_greater_equal(images, -1.)]):
        with tf.control_dependencies([tf.assert_less_equal(images, 1.)]):
            net = images
            if verbose: print('   \033[34min:\033[0m', net.get_shape())
        
    with tf.variable_scope('conv_1x1_net', reuse=reuse):
        with tf.contrib.framework.arg_scope([slim.conv2d], 
                                            stride=1,
                                            kernel_size=[1, 1],
                                            padding='SAME',
                                            normalizer_fn=None,
                                            biases_initializer=tf.zeros_initializer()):
            # one layer case (~ linear transformation constrained in [0, 1]).
            # init to identity ([-1, 1] -> [0, 1]) with sigmoid-like activation
            if bottleneck_dims == 0:
                weight_init = np.eye(3, dtype=np.float32)
                weight_init = np.reshape(weight_init, (1, 1, 3, 3))
                weights_initializer = tf.constant_initializer(weight_init)
                net = slim.conv2d(net, 3, 
                                  activation_fn=lambda x: tf.nn.sigmoid(2.7 * x),
                                  weights_initializer=weights_initializer,
                                  #biases_initializer=biases_initializer,
                                  scope='conv_out')   
                if verbose: print('   \033[34mconv_out:\033[0m', net.get_shape())
                return net         
            # two-layer case.
            # final sigmoid with tanh activation in the middle
            else:               
                weights_initializer = tf.contrib.layers.xavier_initializer()
                # conv 1
                net = slim.conv2d(net, bottleneck_dims, 
                                  activation_fn=tf.nn.tanh,
                                  weights_initializer=weights_initializer,
                                  scope='conv_inner')
                if verbose: print('   \033[34mconv_inner:\033[0m', net.get_shape())                    
                # conv 2
                net = slim.conv2d(net, 3, 
                                  activation_fn=tf.nn.sigmoid,
                                  weights_initializer=weights_initializer,
                                  scope='conv_out')            
                if verbose: print('   \033[34mconv_out:\033[0m', net.get_shape())
                # return
                return net