from functools import partial

import tensorflow as tf

from . import net_utils
from . import viz_utils

"""Preprocessing wrappers and transition from the 
   preprocessed outputs to the classifier.
   Only used for section 5.4 in the paper"""

def mnist_transition(images, is_chief=True):
    """ Transition -> to mnist classifier 
    
    Args:
        images: a 4D Tensor of images
        
    Returns:
        (., 28, 28, 1) grayscaled images in [0., 1.]
    """
    del is_chief
    
    with tf.control_dependencies([tf.assert_greater_equal(images, 0.)]):
        with tf.control_dependencies([tf.assert_less_equal(images, 1.)]): 
            # resize
            size = images.get_shape()[1].value
            if size != 28:
                images = tf.image.resize_nearest_neighbor(images, (28, 28))

            # grayscale
            channels = images.get_shape()[3].value
            if channels != 1:
                images = tf.reduce_mean(images, axis=3, keep_dims=True)
            return images
        
        
def cifar_transition(images, num_channels=3, is_chief=True):
    """ Transition -> to car classifier 
    
    Args:
        images: a 4D Tensor of images
        
    Returns:
        (., 32, 32, 1 or 3)  images in [0., 1.]
    """
    del is_chief
    
    with tf.control_dependencies([tf.assert_greater_equal(images, 0.)]):
        with tf.control_dependencies([tf.assert_less_equal(images, 1.)]): 
            # resize
            size = images.get_shape()[1].value
            if size != 32:
                images = tf.image.resize_nearest_neighbor(images, (32, 32))

            # grayscale
            channels = images.get_shape()[3].value
            assert channels in [1, 3]
            if channels == 1 and num_channels == 3:
                images = tf.tile(images, (1, 1, 1, 3))
            if channels == 3 and num_channels == 1:
                images = tf.reduce_mean(images, axis=3, keep_dims=True)
            return images

        
def imagenet_transition(images, is_chief=True):
    """ Transition -> to inception classifier 
    
    Args:
        images: a 4D Tensor of images
        
    Returns:
        (., 224, 224, 3) images in [0., 1.]
    
    """
    del is_chief  
    
    with tf.control_dependencies([tf.assert_greater_equal(images, 0.)]):
        with tf.control_dependencies([tf.assert_less_equal(images, 1.)]):
            # channels (static)
            channels = images.get_shape()[3].value
            assert channels == 3
            # resize
            size = images.get_shape()[1].value
            if size != 224:
                images = tf.image.resize_nearest_neighbor(images, (224, 224))
            return images

        
def pix2pix_preprocess(images, 
                       num_preprocessing_layers=0, 
                       num_outputs=3,
                       encoder_base_num_filters=32,
                       is_training=True, 
                       reuse=False,
                       is_chief=True,
                       num_rows=5,
                       batch_size=32,
                       verbose=True, 
                       **kwargs): 
    """Free-form transformation preprocessing.
    
    Args:
        x: 4D tensor of images
        num_preprocessing_layers: I negative, number of layers in the pix2pix encoder. 
        num_outputs: Number of output channels for Pix2Pix
        encoder_base_num_filters: Base number of filters in the pix2pix encoder
        num_rows: required for summary (if is_chief is True)
        batch_size: required for summary (if is_chief is True)
        is_training: whether the model is in training mode
        reuse: whether to reuse the model.
        is_chief: determine whether to add summaries
        verbose: verbosity level
        kwargs: Unused keywords arguments
    """
    assert num_preprocessing_layers <= 0
    
    # No preprocessing
    if num_preprocessing_layers == 0:
        return images
    # Pix2pix
    else:
        # Pix2Pix take images in [-1, 1]
        with tf.control_dependencies([tf.assert_greater_equal(images, 0.)]):
            with tf.control_dependencies([tf.assert_less_equal(images, 1.)]):  
                images = (images - 0.5) * 2
            if is_chief:
                input_images = viz_utils.image_grid(images, num_rows=num_rows, batch_size=batch_size)
        
        # Pix2pix. Output in [-1, 1]
        encoder_blocks = [encoder_base_num_filters * (2**i) for i in range(-num_preprocessing_layers)]
        images = net_utils.pix2pix(images,
                                   encoder_blocks=encoder_blocks,
                                   num_outputs=num_outputs, 
                                   is_training=is_training,
                                   reuse=reuse,
                                   is_chief=is_chief,
                                   verbose=verbose,
                                   **kwargs)
        # Image summaries
        if is_chief:
            images = tf.identity(images, name='projected_images')
            mode = ('train' if is_training else 'test')
            output_images = viz_utils.image_grid(images, num_rows=num_rows, batch_size=batch_size)
            # Tile if necessary
            if num_outputs == 1 and input_images.get_shape()[-1] == 3:
                output_images = tf.tile(output_images, (1, 1, 1, 3))
            if input_images.get_shape()[-1] == 1 and num_outputs == 3:
                input_images = tf.tile(input_images, (1, 1, 1, 3))
            # Grid
            summary_images = viz_utils.image_grid(
                tf.concat([input_images, output_images], axis=0), num_rows=1, num_cols=2, batch_size=2)
            summary_images = tf.identity(summary_images, name='in_out_images')
            tf.summary.image('%s/in_out' % mode, summary_images, collections=[mode]) 
            
        # Send output back to [0, 1]
        images = images / 2. + 0.5   
        return images

mnist_preprocess = partial(pix2pix_preprocess, num_outputs=1)

imagenet_preprocess = partial(pix2pix_preprocess, num_outputs=3)


def spatial_preprocess(images, 
                       encoder_blocks,
                       num_preprocessing_layers=0, 
                       is_training=True, 
                       reuse=False,
                       is_chief=True,
                       num_rows=5,
                       batch_size=32,
                       verbose=True, 
                       **kwargs): 
    """Preprocessing net for spatial transformer.
    
    Args:
        x: 4D tensor of images
        encoder_blocks: The number of encoder layers in the STN
        num_preprocessing_layers: I negative, applies the STN, otherwise identity
        encoder_base_num_filters: Base num filters in the pix2pix encoder
        is_training: whether the model is in training mode
        reuse: whether to reuse the model.
        is_chief: determine whether to add summaries
        batch_size: required for summary (if is_chief is True)
        num_rows: required for image summary (if is_chief is True)
        verbose: verbosity level
    """
    assert num_preprocessing_layers in [0, -1]
    # No preprocessing
    if num_preprocessing_layers == 0:
        return images
    # STN
    else:
        if is_chief:
            input_images = viz_utils.image_grid(images, num_rows=num_rows, batch_size=batch_size)
            
        # Spatial transformer
        images = net_utils.spatial_transformer(images, 
                                               encoder_blocks=encoder_blocks,
                                               is_training=is_training,
                                               reuse=reuse, 
                                               is_chief=is_chief,
                                               verbose=verbose,
                                               **kwargs)
        # add summary for chief
        if is_chief:
            images = tf.identity(images, name='projected_images')
            mode = ('train' if is_training else 'test')
            output_images = viz_utils.image_grid(images, num_rows=num_rows, batch_size=batch_size)
            summary_images = viz_utils.image_grid(
                tf.concat([input_images, output_images], axis=0), num_rows=1, num_cols=2, batch_size=2)
            summary_images = tf.identity(summary_images, name='in_out_images')
            tf.summary.image('%s/in_out' % mode, summary_images, collections=[mode])  
        return images

mnist_spatial_preprocess = partial(spatial_preprocess, encoder_blocks=[128])

ilsvrc_spatial_preprocess = partial(spatial_preprocess, encoder_blocks=[32, 64, 128, 64])


def color_preprocess(images,
                     bottleneck_dims=16,
                     num_preprocessing_layers=0, 
                     is_training=True, 
                     reuse=False,
                     is_chief=True,
                     num_rows=5,
                     batch_size=32,
                     verbose=True, 
                     **kwargs): 
    """Preprocessing net for color channels transforms.
    
    Args:
        x: 4D tensor of images
        num_preprocessing_layers: If negative applies the preprocessing with 1 or 2 layers, otherwise identity
        is_training: whether the model is in training mode
        reuse: whether to reuse the model.
        is_chief: determine whether to add summaries
        batch_size: required for summary (if is_chief is True)
        num_rows: required for image summary (if is_chief is True)
        verbose: verbosity level
    """
    assert num_preprocessing_layers in [0, -1, -2]
    # No preprocessing
    if num_preprocessing_layers == 0:
        return images
    else:
        if is_chief:
            input_images = viz_utils.image_grid(images, num_rows=num_rows, batch_size=batch_size)
            
        # Scale images to [-1, 1]
        with tf.control_dependencies([tf.assert_greater_equal(images, 0.)]):
            with tf.control_dependencies([tf.assert_less_equal(images, 1.)]):  
                images = (images - 0.5) * 2
            
        # Preprocessing
        images = net_utils.conv_1x1_net(images, 
                                        bottleneck_dims=bottleneck_dims * int(num_preprocessing_layers == -2),
                                        is_training=is_training,
                                        reuse=reuse, 
                                        is_chief=is_chief,
                                        verbose=verbose,
                                        **kwargs)
        # add summary for chief
        if is_chief:
            images = tf.identity(images, name='projected_images')
            mode = ('train' if is_training else 'test')
            output_images = viz_utils.image_grid(images, num_rows=num_rows, batch_size=batch_size)
            summary_images = viz_utils.image_grid(
                tf.concat([input_images, output_images], axis=0), num_rows=1, num_cols=2, batch_size=2)
            summary_images = tf.identity(summary_images, name='in_out_images')
            tf.summary.image('%s/in_out' % mode, summary_images, collections=[mode])  
        return images