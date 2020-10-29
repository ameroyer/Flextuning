from functools import partial 

import tensorflow as tf

from . import preprocess_utils
from . import tfrecords_utils


def create_tf_dataset(path_to_tfrecords, 
                      parsing_fn, 
                      parsing_fn_kwargs={},
                      batch_size=1, 
                      shuffle_buffer=1,
                      prefetch=1,
                      num_parallel_calls=4,
                      num_epochs=-1,
                      shards=1,
                      index=0,
                      reinitializable=False):  
    """Create a Tensorflow dataset from a TFRecords objects.
    
    Args:
        path_to_tfrecords: path to the TFRecords
        parsing_fn: Parsing function. Has signature
          * example_proto: A proto from the TFRecords
          * kwargs
        parsing_fn_kwargs: Optional dictoinnary of keywords arguments for `parsing_fn`
        batch_size: batch_size
        shuffle_bufer: shuffle buffer
        prefetch: prefetch capacity if the iterator is repeated
        num_parallel_calls: number of readers for the map function (note: nunmber of readers per shard)
        num_epochs: If > 0, repeat the training dataset for the given number of epochs, otherwise infinite repeat.
        shards: If > 1, shards the dataset
        index: Use this worker index, if sharding.
        reinitializable: Whether to make the iterator reinitializable (and non repeated)
    """
    # Read example protos
    data = tf.data.TFRecordDataset(path_to_tfrecords)
    # Shard
    if shards > 1:
        data = data.shard(shards, index)
    # Shufle
    if shuffle_buffer > 1:
        data = data.shuffle(shuffle_buffer)
    # Map
    data = data.map(lambda x: parsing_fn(x, **parsing_fn_kwargs), num_parallel_calls=num_parallel_calls)
    # Make the iterator
    if not reinitializable: 
        if num_epochs > 0:
            data = data.repeat(num_epochs)
        else:
            data = data.repeat()
        data = data.batch(batch_size)
        if prefetch > 0: 
            data = data.prefetch(prefetch)
        iterator = data.make_one_shot_iterator()
    else:
        data = data.batch(batch_size)
        iterator = data.make_initializable_iterator()
    return iterator


def get_dataset(mode,
                parsing_fn,
                parsing_fn_kwargs,
                train_tfrecords, 
                test_tfrecords,
                val_tfrecords=None,
                batch_size=1, 
                shuffle_buffer=1,
                prefetch=1,
                num_parallel_calls=8,
                num_epochs=-1,
                shards=1,
                index=0,
                shuffle_test=False,
                **kwargs):  
    """Get a dataset object for train or test time.
    
    Args:
        mode: One of `train`, `test` or 'viz_train' (visualize the training set)
        train_tfrecords: Path to TFRecords for training
        test_tfrecords: Path to TFRecords for testing
        parsing_fn: Parsing function. Has signature
          * example_proto: A proto from the TFRecords
          * kwargs
        parsing_fn_kwargs: Optional dictoinnary of keywords arguments for the parsing function        
        batch_size: batch_size
        shuffle_bufer: shuffle buffer (always applied to train split, only applied to test if `shuffle_test` is True)
        prefetch: prefetch capacity for the repeated train iterator
        num_parallel_calls: number of readers for the map function
        num_epochs: If > 0, repeat the training dataset for the given number of epochs, otherwise infinite repeat.
        shards: If > 1, shards the dataset
        index: Use this worker index, if sharding.
        shuffle_test: whether to shuffle the test set or not
    """
    
    if mode == 'train':
        tfrecords = train_tfrecords
        reinitializable = False
    elif mode == 'test':
        tfrecords = test_tfrecords
        shuffle_buffer = -1 if not shuffle_test else shuffle_buffer
        reinitializable = True
    elif mode == 'val':
        tfrecords = val_tfrecords
        shuffle_buffer = -1 if not shuffle_test else shuffle_buffer
        reinitializable = True
    elif mode == 'viz_train':
        tfrecords = train_tfrecords
        shuffle_buffer = -1
        reinitializable = True        
    else:
        raise ValueError('Unknown mode', mode)
    assert tfrecords is not None
    return create_tf_dataset(tfrecords,
                             parsing_fn,
                             parsing_fn_kwargs=parsing_fn_kwargs,
                             batch_size=batch_size,
                             shuffle_buffer=shuffle_buffer,
                             prefetch=prefetch,
                             num_parallel_calls=num_parallel_calls,
                             num_epochs=num_epochs,
                             shards=shards,
                             index=index,
                             reinitializable=reinitializable) 

### For computing class ratios, a dataset that only load sample classes
def get_class_only_dataset(mode,
                           train_tfrecords=None, 
                           test_tfrecords=None,
                           class_key='class',
                           **kwargs):
    def parsing_fn(example_proto):
        parsed_features = tf.parse_single_example(example_proto, {class_key: tf.FixedLenFeature((), tf.int64)} )   
        return {'class': tf.to_int32(parsed_features[class_key])}
    return get_dataset(mode, parsing_fn, {}, train_tfrecords, test_tfrecords, **kwargs)


### MNIST
def _get_basic_dataset(mode, 
                       parsing_fn,
                       train_tfrecords=None, 
                       test_tfrecords=None,
                       image_size=None,
                       **kwargs):
    """ Dataset loading function for basic dataset (only need an image_size argument)"""
    parsing_fn_kwargs = {'image_size': image_size}
    return get_dataset(mode, parsing_fn, parsing_fn_kwargs, train_tfrecords, test_tfrecords, **kwargs)
    
get_mnist_dataset = partial(_get_basic_dataset, parsing_fn=tfrecords_utils.mnist_parsing_fn)

get_occluded_mnist_dataset = partial(_get_basic_dataset, parsing_fn=tfrecords_utils.occluded_mnist_parsing_fn)

get_spatial_transformed_mnist_dataset = partial(
    _get_basic_dataset, parsing_fn=tfrecords_utils.spatial_transformed_mnist_parsing_fn)

get_svhn_dataset = partial(_get_basic_dataset, parsing_fn=tfrecords_utils.svhn_parsing_fn)

get_mnistm_dataset = partial(_get_basic_dataset, parsing_fn=tfrecords_utils.mnistm_parsing_fn)

get_cifar_dataset = partial(_get_basic_dataset, parsing_fn=tfrecords_utils.cifar_parsing_fn)

get_quickdraw_dataset = partial(_get_basic_dataset, parsing_fn=tfrecords_utils.quickdraw_parsing_fn)


### ILSVRC and PACS
def _get_basic_relative_dataset(mode, 
                                parsing_fn,
                                train_tfrecords=None, 
                                test_tfrecords=None, 
                                image_size=None,
                                crop_size=None,
                                image_dir=None,
                                **kwargs):
    """ Dataset loading function for datasets loaded from files"""
    assert image_dir is not None
    assert (crop_size is None) or (image_size is not None)
    parsing_fn_kwargs = {'image_size': image_size, 'crop_size': crop_size, 'image_dir': image_dir}
    return get_dataset(mode, parsing_fn, parsing_fn_kwargs, train_tfrecords, test_tfrecords, **kwargs)

get_pacs_dataset = partial(_get_basic_relative_dataset, parsing_fn=tfrecords_utils.pacs_parsing_fn)

get_ilsvrc_dataset = partial(_get_basic_relative_dataset, parsing_fn=tfrecords_utils.ilsvrc_parsing_fn)

get_occluded_ilsvrc_dataset = partial(_get_basic_relative_dataset, parsing_fn=tfrecords_utils.occluded_ilsvrc_parsing_fn)

get_transformed_ilsvrc_dataset = partial(_get_basic_relative_dataset, 
                                         parsing_fn=tfrecords_utils.spatial_transformed_ilsvrc_parsing_fn)


### Additional on-the-fly pre-processing
def get_blurry_dataset(mode,
                       parsing_fn,
                       train_tfrecords=None, 
                       test_tfrecords=None, 
                       image_size=None,
                       crop_size=None,
                       image_dir=None,
                       kern_len=8,
                       kern_sig=2,
                       intermediate_image_size=None,
                       **kwargs):
    assert (crop_size is None) or (image_size is not None and intermediate_image_size is not None)
    gauss_kernel = preprocess_utils.gauss_kernel(kernlen=kern_len, nsig=kern_sig)  
    if intermediate_image_size is None:
        intermediate_image_size = image_size
        image_size = None
    parsing_fn_kwargs = {'intermediate_image_size': intermediate_image_size, 
                         'image_size': image_size, 
                         'gauss_kernel': gauss_kernel} 
    if image_dir is not None:
        parsing_fn_kwargs['image_dir'] = image_dir
    if crop_size is not None:
        parsing_fn_kwargs['image_size'] = crop_size
        parsing_fn_kwargs['crop_size'] = int(crop_size * intermediate_image_size / image_size)
    blurry_parsing_fn = partial(tfrecords_utils.blur, parsing_fn=parsing_fn)
    return get_dataset(mode, blurry_parsing_fn, parsing_fn_kwargs, train_tfrecords, test_tfrecords, **kwargs)

get_blurry_mnist_dataset = partial(get_blurry_dataset, parsing_fn=tfrecords_utils.mnist_parsing_fn)

get_blurry_cifar_dataset = partial(get_blurry_dataset, parsing_fn=tfrecords_utils.cifar_parsing_fn)

get_blurry_quickdraw_dataset = partial(get_blurry_dataset, parsing_fn=tfrecords_utils.quickdraw_parsing_fn)

get_blurry_ilsvrc_dataset = partial(get_blurry_dataset, parsing_fn=tfrecords_utils.ilsvrc_parsing_fn)


def get_fixed_downscaling_ilsvrc_dataset(mode,
                                         train_tfrecords=None, 
                                         test_tfrecords=None, 
                                         image_size=None,
                                         crop_size=None,
                                         image_dir=None,
                                         scale=2.0,
                                         padding='EDGE',
                                         **kwargs):
    parsing_fn_kwargs = {'scale': scale, 'padding': padding, 'image_size': image_size, 'crop_size': crop_size,
                         'image_dir': image_dir}        
    parsing_fn = partial(tfrecords_utils.fixed_downscaling, parsing_fn=tfrecords_utils.ilsvrc_parsing_fn)
    return get_dataset(mode, parsing_fn, parsing_fn_kwargs, train_tfrecords, test_tfrecords, **kwargs)


def get_fixed_rotation_ilsvrc_dataset(mode,
                                      train_tfrecords=None, 
                                      test_tfrecords=None, 
                                      image_size=None,
                                      crop_size=None,
                                      image_dir=None,
                                      angle=0.,
                                      padding='EDGE',
                                      **kwargs):
    parsing_fn_kwargs = {'angle': angle, 'image_size': image_size, 'crop_size': crop_size, 'image_dir': image_dir}        
    parsing_fn = partial(tfrecords_utils.fixed_rotation, parsing_fn=tfrecords_utils.ilsvrc_parsing_fn)
    return get_dataset(mode, parsing_fn, parsing_fn_kwargs, train_tfrecords, test_tfrecords, **kwargs)


get_hsv_ilsvrc_dataset = partial(_get_basic_relative_dataset, 
                                 parsing_fn=partial(tfrecords_utils.hsv_ize, parsing_fn=tfrecords_utils.ilsvrc_parsing_fn))

get_yuv_ilsvrc_dataset = partial(_get_basic_relative_dataset, 
                                 parsing_fn=partial(tfrecords_utils.yuv_ize, parsing_fn=tfrecords_utils.ilsvrc_parsing_fn))