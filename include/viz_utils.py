import os
from functools import partial
from collections import defaultdict

from matplotlib import cm
from matplotlib import pyplot as plt

import numpy as np
import tensorflow as tf


def get_grid_size(n):
    """Return square-ish grid layout to fit `n` elements"""
    num_rows = 1
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            num_rows = i
    num_cols = n // num_rows
    return {'num_rows': num_rows, 'num_cols': num_cols}


def image_grid(images, num_rows=4, num_cols=None, batch_size=None, padding_h=1, padding_v=1):
    """Stack images into a square grid with the given number of rows.
    
    Args:
      images: 4D Tensor of images with shape (b, w, h, c).
      num_rows: Number of rows in the grid.
      num_cols: Number of columns in the grid.
      batch_size: Number of images in `images`
      
    Returns:
        A Tensor of shape (num_rows * w, num_cols * h, c)
    """
    # 12 12
    num_cols = num_rows
    num_rows = 1
    # 12 12
    assert batch_size is not None
    # set grid if non specificied
    grid_size = get_grid_size(batch_size)  
    if num_rows is None and num_cols is None:
        num_rows = grid_size['num_rows']
        num_cols = grid_size['num_cols']
    # otherwise check that the specified grid fits
    else:
        if num_cols is None: num_cols = num_rows
        if num_rows is None: num_rows = num_cols
        if grid_size['num_rows'] < num_rows or grid_size['num_cols'] < num_cols:
            num_rows = grid_size['num_rows']
            num_cols = grid_size['num_cols'] 
    # Make the grid
    images = images[:num_rows * num_cols]
    images = tf.pad(images, ((0, 0), (padding_h, padding_h), (padding_v, padding_v), (0, 0)), constant_values=1.)
    images = tf.unstack(images, num=num_rows * num_cols, axis=0) 
    images = tf.concat(images, axis=1) 
    images = tf.split(images, num_rows, axis=1) 
    images = tf.concat(images, axis=0) 
    return tf.expand_dims(images, 0)


def add_image_summary(name, images, batch_size, num_rows=5, num_cols=None, collections=['train']):
    """Add an image grid summary.
    
    Ars:
        name: summary name
        images: 4D Tensor of images
        batch_size: Integer specifying the batch size
        num_rows: Number of rows in the grid
        num_cols: Number of columns in the grid
        collections: Collections to add the summaries to
    """ 
    tf.summary.image(name, image_grid(images, num_rows=num_rows, num_cols=num_cols, batch_size=batch_size), 
                     max_outputs=1, collections=collections)
    
    
def display_dataset(get_dataset_fn,
                    mode='train', 
                    num_displays=8, 
                    figwidth=16,
                    gpu_memory_fraction=0.1,
                    config={},
                    verbose=False):
    """Display a few samples of the given dataset of TFRecords.
    
    Args:
        get_dataset_fn: a function from dataset_utils.
        mode: One of `train`, 'val' or `test`. Select which split to display.
        num_displays: Number of samples to display.
        figwidth: Width of the matplotlib figure.
        gpu_memory_fraction: Fraction of GPU memory to use
        config: Addictional keyword configuration given to the dataset function
        verbose: If verbose, count the whole dataset
    """
    assert mode in ['train', 'test', 'val']
    # Init the dataset
    with tf.Graph().as_default():
        inputs = get_dataset_fn('viz_train' if mode == 'train' else mode, **config)
        init = inputs.initializer
        inputs = inputs.get_next()
        if verbose:  print('\n'.join('   \033[36m%s\033[0m: %s' % (key, v.get_shape()) for key, v in inputs.items()))
            
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_memory_fraction))) as sess:
            num_samples_per_class = defaultdict(lambda: 0)
            sess.run(init)
            try:
                # First iter: display the batch
                inputs_ = sess.run(inputs)
                num_displays = min(num_displays, inputs_['class'].shape[0])
                fig, axis = plt.subplots(1, num_displays, figsize=(figwidth, figwidth // num_displays ))
                if num_displays == 1:
                    axis = [axis]
                for j in range(num_displays):
                    img = inputs_['image'][j]
                    if img.shape[-1] == 1: img = np.tile(img, (1, 1, 3))
                    axis[j].imshow(img)
                    if 'class' in inputs_:
                        if 'class_name' in inputs_:
                            axis[j].set_title('Class %d\n%s' % (
                                inputs_['class'][j], inputs_['class_name'][j].decode('utf-8').split(',')[0]), fontsize=8)  
                        else:
                            axis[j].set_title('Class %d' % inputs_['class'][j], fontsize=10)                        
                    axis[j].set_axis_off()
                    plt.subplots_adjust(wspace=0.02)
                # If verbose: count the whole dataset
                if verbose:
                    for c in inputs_['class']:
                        num_samples_per_class[c] += 1
                    while 1:
                        inputs_ = sess.run(inputs)
                        for c in inputs_['class']:
                            num_samples_per_class[c] += 1
            except tf.errors.OutOfRangeError:
                pass                   
    if verbose:
        print('   \033[31m%d\033[0m samples in dataset' % sum(num_samples_per_class.values()))
        print('  ', ', '.join('\033[35m%d\033[0m (%d)' % (c, v) for c, v in num_samples_per_class.items()))
    plt.show()