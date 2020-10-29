from collections import defaultdict
from functools import partial

import numpy as np
import tensorflow as tf

from . import preprocess_utils


""" tf.data utils

Defines functions related to handlin the TFRecords inputs:
  * Subsampling of datasets (while respecting class ratios)
  * Various input loading functions for the different datasets
  * Input transformation functions to create the target domains (e.g. blur)
"""


def get_class_ratios(get_dataset_fn, config, seed=None, verbose=True):
    """Compute a class ratio for each sample in the given train dataset, shuffled. 
    This can then be used to subsample from the dataset.
    
    Args:
        get_dataset_fn: A function from dataset_utils to load a dataset. The `viz_train` mode will 
            be used (i.e., a one-pass iterator on the unshuffled training dataset)
        config: Configuration dictionnary to be fed to `get_dataset_fn`
        seed: If given, fix a random seed for the shuffle operation
        verbose: Verbosity level
            
    """
    # Get class ratios
    # samples_per_class: For each class, we collect indices of its elements (in the initial unsorted dataset)
    with tf.Graph().as_default():
        data = get_dataset_fn('viz_train', **config)
        inputs = data.get_next()
        labels = inputs['class']
        samples_per_class = defaultdict(lambda : [])
        with  tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1))) as sess:
            sess.run(data.initializer)
            try:
                sample_index = 0
                while 1:
                    labels_ = sess.run(labels)
                    for label in labels_:
                        samples_per_class[label].append(sample_index)
                        sample_index += 1
            except tf.errors.OutOfRangeError:
                pass
            
    # Shuffle
    # class_ratios: For each sample, compute its position (ratio) in its respective class
    num_samples = sum(len(x) for x in samples_per_class.values())
    class_ratios = [0.] * num_samples
    if seed is not None:
        state = np.random.get_state()
        np.random.seed(seed)
    for label, lst in samples_per_class.items():
        np.random.shuffle(lst)
        num_samples_class = len(lst)
        for pos_in_class, index in enumerate(lst):
            class_ratios[index] = pos_in_class / num_samples_class
    if seed is not None:
        np.random.set_state(state)
        
    # Display
    if verbose:
        print('   %d samples' % num_samples)
        print('  ', ', '.join('%d \033[34m(%d)\033[0m' % (len(samples_per_class[key]), key)
                               for key in sorted(samples_per_class.keys())))     
    return class_ratios


def decode_raw_image(feature, shape, image_size=None):
    """Decode raw image
    Args:
        feature: raw image as a tf.String tensor
        shape: Shape of the raw image
        image_size: If given, resize the decoded image to the given square size
    Returns:
        The resized image
    """
    image = tf.decode_raw(feature, tf.uint8)
    image = tf.reshape(image, shape)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if image_size is not None:
        image = tf.image.resize_images(image, (image_size, image_size))
    return image


def decode_relative_image(feature, image_dir, image_size=None):
    """Decode image from a filename
    Args:
        feature: image path as a tf.String tensor
        image_dir: Nase image dir
        image_size: If given, resize the decoded image to the given square size
    Returns:
        The resized image
    """
    filename = tf.decode_base64(feature)
    image = tf.read_file(image_dir + filename)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if image_size is not None:
        image = tf.image.resize_images(image, (image_size, image_size))
    return image


########################################################################## MNIST
def mnist_parsing_fn(example_proto, shape=(28, 28, 1), image_size=None):
    """MNIST parsing function."""
    features = {'class': tf.FixedLenFeature((), tf.int64),
                'image': tf.FixedLenFeature((), tf.string)}      
    parsed_features = tf.parse_single_example(example_proto, features)  
    image = decode_raw_image(parsed_features['image'], shape, image_size=image_size)
    class_id = tf.to_int32(parsed_features['class'])
    return {'image': image, 'class': class_id}

mnistm_parsing_fn = partial(mnist_parsing_fn, shape=(28, 28, 3))

svhn_parsing_fn = partial(mnist_parsing_fn, shape=(32, 32, 3))


def occluded_mnist_parsing_fn(example_proto, image_size=None):
    """Occluded MNIST parsing function. Create a black patch from the coordinates given in the TFRecords.
    """
    features = {'class': tf.FixedLenFeature((), tf.int64),
                'image': tf.FixedLenFeature((), tf.string),
                'patch_coords': tf.FixedLenFeature((4), tf.float32)}      
    parsed_features = tf.parse_single_example(example_proto, features)  
    image = decode_raw_image(parsed_features['image'], (28, 28, 1), image_size=image_size)
    class_id = tf.to_int32(parsed_features['class'])  
    # occlusion
    patch_coords = parsed_features['patch_coords']
    image = preprocess_utils.apply_occlusion_mask(image, patch_coords)    
    return {'image': image, 'class': class_id}   


def spatial_transformed_mnist_parsing_fn(example_proto, image_size=None):
    """Spatially transformed MNIST parsing function. 
    Apply an affine transformation with the parameters given in the TFrecords
    """
    features = {'class': tf.FixedLenFeature((), tf.int64),
                'image': tf.FixedLenFeature((), tf.string),
                'transform': tf.FixedLenFeature((9,), tf.float32)}   
    parsed_features = tf.parse_single_example(example_proto, features)  
    image = decode_raw_image(parsed_features['image'], (28, 28, 1), image_size=image_size)
    class_id = tf.to_int32(parsed_features['class'])   
    # Transform
    transform_matrix = parsed_features['transform']
    transform_matrix = tf.reshape(transform_matrix, (-1,))[:8]
    transform_matrix = tf.reshape(transform_matrix, (1, 8))
    image = tf.contrib.image.transform(image, transform_matrix, interpolation='NEAREST')    
    image = tf.clip_by_value(image, 0., 1.)
    return {'image': image, 'class': class_id}  


########################################################################## PACS
def pacs_parsing_fn(example_proto, crop_size=None, image_size=None, image_dir=''):
    """PACS parsing function. Load image from image_dir and resize them to `image_size`"""
    features = {'class_content': tf.FixedLenFeature((), tf.int64),
                'image': tf.FixedLenFeature((), tf.string)}      
    parsed_features = tf.parse_single_example(example_proto, features)  
    image = decode_relative_image(parsed_features['image'], image_dir, image_size=image_size)
    class_id = tf.to_int32(parsed_features['class_content'])
    return {'image': image, 'class': class_id}


########################################################################## CIFAR
def cifar_parsing_fn(example_proto, shape=(32, 32, 3), image_size=None):
    """CIFAR parsing function."""
    features = {'class': tf.FixedLenFeature((), tf.int64),
                'image': tf.FixedLenFeature((), tf.string),
                'class_name': tf.FixedLenFeature((), tf.string)} 
    parsed_features = tf.parse_single_example(example_proto, features)  
    image = decode_raw_image(parsed_features['image'], shape, image_size=image_size)
    class_id = tf.to_int32(parsed_features['class'])
    class_name = tf.decode_base64(parsed_features['class_name'])
    return {'image': image, 'class': class_id, 'class_name': class_name}

quickdraw_parsing_fn = partial(cifar_parsing_fn, shape=(28, 28, 1))


##########################################################################  ImageNet
def central_crop(image, image_size, crop_size):
    width = tf.shape(image)[0]
    height = tf.shape(image)[1]
    min_side = tf.minimum(width, height)
    target_width = tf.to_int32(image_size * width / min_side)
    target_height = tf.to_int32(image_size * height / min_side)
    image = tf.image.resize_images(image, (target_width, target_height))
    image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
    image = tf.reshape(image, (crop_size, crop_size, -1))
    return image
    
def ilsvrc_parsing_fn(example_proto, image_size=None, crop_size=None, image_dir=''):
    """ILSVRC parsing function. Load image from image_dir and resize them to `image_size`"""
    features = {'image': tf.FixedLenFeature((), tf.string),
                'class': tf.FixedLenFeature((), tf.int64),
                'class_name': tf.FixedLenFeature((), tf.string)}     
    parsed_features = tf.parse_single_example(example_proto, features) 
    if crop_size is None or image_size is None:
        image = decode_relative_image(parsed_features['image'], image_dir, image_size=image_size)
    else:
        image = decode_relative_image(parsed_features['image'], image_dir, image_size=None)
        image = central_crop(image, image_size, crop_size)
    class_id = tf.to_int32(parsed_features['class'])
    class_name = tf.decode_base64(parsed_features['class_name'])
    return {'image': image, 'class': class_id, 'class_name': class_name}


def occluded_ilsvrc_parsing_fn(example_proto, image_size=None, crop_size=None, image_dir=''):
    """Occluded ILSVRC parsing function. Create a black patch from the coordinates given in the TFRecords.
    """
    features = {'class': tf.FixedLenFeature((), tf.int64),
                'image': tf.FixedLenFeature((), tf.string),
                'class_name': tf.FixedLenFeature((), tf.string),
                'patch_coords': tf.FixedLenFeature((4), tf.float32)}     
    parsed_features = tf.parse_single_example(example_proto, features) 
    if crop_size is None or image_size is None:
        image = decode_relative_image(parsed_features['image'], image_dir, image_size=image_size)
    else:
        image = decode_relative_image(parsed_features['image'], image_dir, image_size=None)
        image = central_crop(image, image_size, crop_size)
    class_id = tf.to_int32(parsed_features['class'])
    class_name = tf.decode_base64(parsed_features['class_name'])
    # occlusion
    patch_coords = parsed_features['patch_coords']
    image = preprocess_utils.apply_occlusion_mask(image, patch_coords)    
    return {'image': image, 'class': class_id, 'class_name': class_name}


def spatial_transformed_ilsvrc_parsing_fn(example_proto, image_size=None, crop_size=None, image_dir=''):
    """Spatially transformed ILSVRC parsing function. 
    Apply an affine transformation with the parameters given in the TFrecords
    """
    del crop_size
    features = {'class': tf.FixedLenFeature((), tf.int64),
                'class_name': tf.FixedLenFeature((), tf.string),
                'image': tf.FixedLenFeature((), tf.string),
                'transform': tf.FixedLenFeature((9,), tf.float32)}   
    parsed_features = tf.parse_single_example(example_proto, features)  
    image = decode_relative_image(parsed_features['image'], image_dir, image_size=image_size)
    class_id = tf.to_int32(parsed_features['class'])   
    class_name = tf.decode_base64(parsed_features['class_name'])
    # Transform
    transform_matrix = parsed_features['transform']
    transform_matrix = tf.reshape(transform_matrix, (-1,))[:8]
    transform_matrix = tf.reshape(transform_matrix, (1, 8))
    image = tf.contrib.image.transform(image, transform_matrix, interpolation='NEAREST')    
    image = tf.clip_by_value(image, 0., 1.)
    return {'image': image, 'class': class_id, 'class_name': class_name}


##########################################################################  On-the-fly Transformations
def blur(example_proto, 
         gauss_kernel=None,
         intermediate_image_size=None,
         image_size=None,
         parsing_fn=None,
         **parsing_fn_kwargs):
    """Blurring preprocessing wrapper
    
    Args:
        gauss_kernel: Blur kernel
        intermediate_image_size: Resize the image before blurring them
        image_size: Final size of the image
        parsing_fn: Initial parsing function. Should output an `image` field
        parsing_fn_kwargs: arguments to feed the parsing function
    """    
    assert parsing_fn is not None  
    parsed_output = parsing_fn(example_proto, image_size=intermediate_image_size, **parsing_fn_kwargs)
    image = tf.expand_dims(parsed_output['image'], axis=0)        
    output_channels = image.get_shape()[-1].value
    image = tf.split(image, output_channels, axis=-1)
    with tf.device('/cpu:0'):
        image = tf.concat([tf.nn.conv2d(image[i], gauss_kernel, [1, 1, 1, 1], padding='SAME', data_format='NHWC')[0]
                           for i in range(output_channels)], axis=-1)
    image = tf.clip_by_value(image, 0., 1.)
    # Final resize
    if image_size is not None:
        image = tf.image.resize_images(image, (image_size, image_size))
    parsed_output['image'] = image
    return parsed_output


def noise(example_proto, 
          noise_stddev=0.0,
          parsing_fn=None,
          **parsing_fn_kwargs):
    """White noise preprocessing wrapper
    
    Args:
        noise_stddev: Standard deviation to generate the Gaussian noise
        parsing_fn: Initial parsing function. Should output an `image` field
        parsing_fn_kwargs: arguments to feed the parsing function
    """    
    assert parsing_fn is not None
    parsed_output = parsing_fn(example_proto, **parsing_fn_kwargs)
    image = parsed_output['image']      
    noise = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=noise_stddev, dtype=tf.float32)
    image += noise
    image = tf.clip_by_value(image, 0., 1.)
    parsed_output['image'] = image
    return parsed_output


def fixed_downscaling(example_proto, 
                      scale=1.,
                      parsing_fn=None,
                      padding='EDGE',
                      **parsing_fn_kwargs):
    """White noise preprocessing wrapper
    
    Args:
        scale: The downscaling ratio, msut be > 1.
        parsing_fn: Initial parsing function. Should output an `image` field
        image_size: The final image size
        parsing_fn_kwargs: arguments to feed the parsing function
    """    
    assert parsing_fn is not None
    assert padding in ['EDGE', 'SYMMETRIC']
    assert scale > 1.
    parsed_output = parsing_fn(example_proto, **parsing_fn_kwargs)
    image = parsed_output['image']
    width = image.get_shape()[0] # Note: Need defined shape in the STN
    height = image.get_shape()[1] 
    offset_w = tf.to_int32(tf.to_float(width) * (scale - 1) / 2.)
    offset_h = tf.to_int32(tf.to_float(height) * (scale - 1) / 2.)
    if padding == 'EDGE': 
        # Hack: accurate for imagenet (256 -> 224 crop) scale 2.0 
        pad = 1
        for _ in range(7):
            image = tf.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="SYMMETRIC")
            pad *= 2
    else:
        paddings = tf.stack([tf.stack([offset_w, offset_w], axis=0),
                             tf.stack([offset_h, offset_h], axis=0),
                             tf.stack([0, 0], axis=0)], axis=0)
        image = tf.pad(image, paddings, mode="SYMMETRIC") 
    image = tf.image.resize_images(image, (width, height))
    image = tf.reshape(image, (width, height, 3))
    parsed_output['image'] = image
    return parsed_output


def fixed_rotation(example_proto, 
                   angle=0.,
                   crop_frac=0.75,
                   parsing_fn=None,
                   image_size=None,
                   **parsing_fn_kwargs):
    """White noise preprocessing wrapper
    
    Args:
        scale: The downscaling ratio, msut be > 1.
        parsing_fn: Initial parsing function. Should output an `image` field
        image_size: The final image size
        parsing_fn_kwargs: arguments to feed the parsing function
    """    
    assert parsing_fn is not None
    assert image_size is not None
    parsed_output = parsing_fn(example_proto, image_size=image_size, **parsing_fn_kwargs)
    image = parsed_output['image']     
    width = image.get_shape()[0] # Note: Need defined shape in the STN
    height = image.get_shape()[1] 
    # rotate
    pad = int((1. - crop_frac) * image_size)
    image = tf.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="SYMMETRIC")
    image = tf.contrib.image.rotate(image, angle)
    image = tf.image.central_crop(image, crop_frac)
    image = tf.image.resize_images(image, (width, height))
    parsed_output['image'] = image
    return parsed_output


def colorize(example_proto,
             colorize_fn=None,
             parsing_fn=None,
             **parsing_fn_kwargs):
    """Colorization preprocessing wrapper
    
    Args:
        colorize_fn: Colorization function; takes as input 3D Tensor in [0., 1.] and outputs the same.
        parsing_fn: Initial parsing function. Should output an `image` field
        parsing_fn_kwargs: arguments to feed the parsing function
    """    
    assert colorize_fn is not None
    assert parsing_fn is not None
    parsed_output = parsing_fn(example_proto, **parsing_fn_kwargs)
    image = parsed_output['image']        
    output_channels = image.get_shape()[-1].value
    assert output_channels == 3
    image = colorize_fn(image)
    image = tf.clip_by_value(image, 0., 1.)
    parsed_output['image'] = image
    return parsed_output
   
hsv_ize = partial(colorize, colorize_fn=tf.image.rgb_to_hsv)

yuv_ize = partial(colorize, colorize_fn=preprocess_utils.rgb_to_yuv)