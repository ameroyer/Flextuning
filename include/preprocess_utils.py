import numpy as np
import scipy.stats as st
import tensorflow as tf


""" A few low-level utils functions for the input pipeline. 
    Used in tfrecords_utils.py
"""


def split_tf_records(src_path, tgt_path, skip=0, take=-1, shuffle=False, seed=None, num_elements=None):
    """Split a tfrecords, prealably shuffled optionally.
    Args:
        src_path: tfrecords source path
        tgt_path: tfrecords target path
        skip: How many elements to skip
        take: How many elements to take. Take all the remaining elements if -1.
        shuffle: Whether to apply a shuffle on the sampled indices instead of taking consecutive elements.
        seed: Fix a random seed for shuffling if given
        num_elements: Number of elements in the dataset, should be given if shuffle is true
    Returns:
        Save the tfrecords src_path[skip:skip+take] in tgt_path.
    """
    writer = tf.python_io.TFRecordWriter(tgt_path)
    # shuffled
    if shuffle:
        assert num_elements is not None
        indices = np.arange(num_elements)
        if seed is not None:
            state = np.random.get_state()
            np.random.seed(seed)
        np.random.shuffle(indices)
        if seed is not None:
            np.random.set_state(state)
        for i, example in enumerate(tf.python_io.tf_record_iterator(src_path)):
            if indices[i] >= skip and (take == -1 or indices[i] < skip + take):
                writer.write(example)
    # consecutive
    else:
        for i, example in enumerate(tf.python_io.tf_record_iterator(src_path)):
            if i >= skip and (take == -1 or i < skip + take):
                writer.write(example)
    writer.close()

    
def create_occlusion_masks(src_path, 
                           tgt_path,
                           patch_size=0.5, 
                           seed=42):
    """Generate mask for occluded mnist and add them to the src TFRecords
    
    Args:
        src_path: Source TFRecords
        tgt_path: Target TFRecords
        patch_size: Size of the patches
        seed: Random seed    
    """    
    assert 0. < patch_size < 1.
    assert seed is not None
    state = np.random.get_state()
    np.random.seed(seed)
    try:        
        writer = tf.python_io.TFRecordWriter(tgt_path)        
        for i, example in enumerate(tf.python_io.tf_record_iterator(src_path)):
            patch_start = np.random.uniform(size=(2,), low=0.0, high=1. - patch_size)
            patch_end = patch_start + patch_size
            patch = np.concatenate([patch_start, patch_end])
            new_example = tf.train.Example(features=tf.train.Features(feature={
                'patch_coords': tf.train.Feature(float_list=tf.train.FloatList(value=patch))}))  
            new_example.MergeFromString(example)
            writer.write(new_example.SerializeToString())
    finally:
        np.random.set_state(state)
        
def apply_occlusion_mask(image, patch_coords):
    """Apply an occlusion patch to an image from its coordinates.
    
    Args:
        image: A 3D Tensor of image
        patch_coords: Coordinates of the patch in [0, 1], with format [xmin, ymin, xmax, ymax]
    """
    # Scale patch coordinates
    width, height = image.get_shape().as_list()[:2]
    image_shape = tf.stack([width, height], axis=0)
    patch_coords *= tf.to_float(tf.concat([image_shape] * 2, axis=0))
    patch_coords = tf.to_int32(patch_coords)
    # Pad the patch
    patch_shape = tf.stack((patch_coords[2] - patch_coords[0],
                            patch_coords[3] - patch_coords[1], 1), axis=0)
    patch = tf.zeros(patch_shape)
    paddings = tf.stack([patch_coords[0], width - patch_coords[2], patch_coords[1], height - patch_coords[3], 0, 0], axis=0)
    paddings = tf.reshape(paddings, (3, 2))
    patch = tf.pad(patch, paddings, constant_values=1.)
    # Mask the image
    patch = tf.reshape(patch, (width, height, 1))
    image = patch * image
    return image
        
        
def create_affine_transformation(src_path, 
                                 tgt_path,
                                 center,
                                 min_angle=-1.5,
                                 max_angle=1.5,
                                 min_scale=1.,
                                 max_scale=2.5,
                                 fixed_transform=False,
                                 seed=42):
    """Generate 6-parameters affine transformation of the given dataset. to be used with the tf.contrib.transform function.
    
    Args:
        src_path: Source TFRecords
        tgt_path: Target TFRecords
        center: Coordinates of the center for the rotation and scaling
        min_angle: minimum angle to sample
        max_angle: maximum angle to sample
        min_scale: maximum scaling factor to sample
        max_scale: maximum scaling factor to sample
        fixed_transform: whether to use a fixed transform or sample it for each image
        seed: Random seed
    
    """    
    def sample_transform():
        # shift off-center        
        shift_neg_matrix = np.array([[1, 0, -center[0]], [0, 1, -center[1]], [0, 0, 1]], dtype=np.float32)
        shift_pos_matrix = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]], dtype=np.float32)
        # rotate
        theta = np.random.uniform(low=min_angle, high=max_angle)
        rot = np.array([[np.cos(theta), - np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]], dtype=np.float32)
        # scale
        sx = np.random.uniform(low=min_scale, high=max_scale)
        sy = np.random.uniform(low=min_scale, high=max_scale)
        scale = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float32)
        # translate
        tx = (1. - sx) / 2.
        tx = np.random.uniform(low=-tx, high=tx)
        ty = (1. - sy) / 2.
        ty = np.random.uniform(low=-ty, high=ty)
        translate = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
        # compose
        transform_matrix = np.matmul(
            shift_pos_matrix, np.matmul(translate, np.matmul(scale, np.matmul(rot, shift_neg_matrix))))
        return transform_matrix.flatten()
    
    state = np.random.get_state()
    np.random.seed(seed)
    try:        
        writer = tf.python_io.TFRecordWriter(tgt_path)      
        if fixed_transform:
            transform = sample_transform()
        for i, example in enumerate(tf.python_io.tf_record_iterator(src_path)):
            if not fixed_transform:
                transform = sample_transform()
            new_example = tf.train.Example(features=tf.train.Features(feature={
                'transform': tf.train.Feature(float_list=tf.train.FloatList(value=transform))}))  
            new_example.MergeFromString(example)
            writer.write(new_example.SerializeToString())
    finally:
        np.random.set_state(state)
        
        
def gauss_kernel(kernlen=21, nsig=3):
    """Create a Gaussian kernel [kernlen, kernlen, 1, 1]
    
    Args:
        `kernlen`: Kernel size
        `nsig`: Amplitude
    """
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    return out_filter


def rgb_to_yuv(images):  
    """Convert a 3D image tensor from RGB to YUV (in and output in [0., 1.])"""
    _rgb_to_yuv_kernel = [[0.299, -0.14714119, 0.61497538],
                          [0.587, -0.28886916, -0.51496512],
                          [0.114, 0.43601035, -0.10001026]]
    ndims = images.get_shape().ndims
    images = tf.tensordot(images, _rgb_to_yuv_kernel, axes=[[ndims - 1], [0]])
    # Output in [0., 1] x [-0.43, 0.43] x [-0.61, 0.61]
    # Normalize in [0, 1]
    Y = images[:, :, 0]
    U = (images[:, :, 1] / 0.43601035) * 0.5 + 0.5
    V = (images[:, :, 2] / 0.61497538) * 0.5 + 0.5
    return images