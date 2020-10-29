import os
from functools import partial

from include import dataset_utils
from include import graph_manager
from include import preprocess_utils


""" settings.py
In this script we define the Setting object for each domain shift.
See `input_pipeline.ipynb` to generate initial TFRecords for the base datasets (MNIST, PACS, ILSVRC)

Additionally, running this file as main
    python3 settings.py --[dataset]
generates required TFRecords for some domain shifts (e.g. occlusion) and subsampled training set. 
This should only be run once to ensure that all methods later use the same training, validation and test sets.

A setting is always defined by:
    - a name
    - base_name for the train tfrecords (tf_%s_train)
    - base_name for the val tfrecords (tf_%s_val)
    - base_name for the test tfrecords (tf_%s_test)
    - A TFrecords parsing function defined in `dataset_utils`
"""

ILSVRC_BASE_PATH = os.path.expanduser('~/Datasets/ILSVRC2012/')
PACS_BASE_PATH = os.path.expanduser('~/Datasets/PACS/')
SKETCHY_BASE_PATH = os.path.expanduser('~/Datasets/Sketchy/rendered_256x256/256x256/')

############################################ MNIST domain shifts
# Source
base_mnist_setting = graph_manager.Setting(
    'mnist', 'mnist0-25', 'mnist', 'mnist', dataset_utils.get_mnist_dataset)

# Target domains
blurry_mnist_setting = graph_manager.Setting(
    'blurry_mnist', 'mnist25-60', 'mnist', 'mnist', 
    partial( dataset_utils.get_blurry_mnist_dataset, kern_len=8, kern_sig=1))

occluded_mnist_setting = graph_manager.Setting(
    'occluded_mnist', 'occluded_mnist25-60', 'occluded_mnist', 
    'occluded_mnist', dataset_utils.get_occluded_mnist_dataset)

mnistm_setting = graph_manager.Setting(
    'mnist-m', 'mnist-m25-60', 'mnist-m', 
    'mnist-m', dataset_utils.get_mnistm_dataset)

svhn_setting = graph_manager.Setting(
    'svhn', 'svhn', 'svhn', 'svhn', dataset_utils.get_svhn_dataset)

transformed_mnist_setting = graph_manager.Setting(
    'transformed_mnist', 'transformed_mnist25-60', 'transformed_mnist',  
    'transformed_mnist', dataset_utils.get_spatial_transformed_mnist_dataset)

fixed_transformed_mnist_setting = graph_manager.Setting(
    'fixed_transformed_mnist', 'fixed_transformed_mnist25-60',
    'fixed_transformed_mnist', 'fixed_transformed_mnist',
    dataset_utils.get_spatial_transformed_mnist_dataset)


############################################## CIFAR & QUICKDRAW domain shifts
# Source
cifar_setting = graph_manager.Setting(
    'cifar', 'cifar', 'cifar', 'cifar', dataset_utils.get_cifar_dataset)

# Target domains
blurry_cifar_setting = graph_manager.Setting(
    'blurry_cifar', 'cifar', 'cifar', 'cifar', 
    partial(dataset_utils.get_blurry_cifar_dataset, kern_len=4, kern_sig=0.5))

noisy_cifar_setting = graph_manager.Setting(
    'noisy_cifar', 'noisy_cifar', 'noisy_cifar', 'noisy_cifar', 
    dataset_utils.get_cifar_dataset)

quickdraw_setting = graph_manager.Setting(
    'quickdraw', 'quickdraw', 'quickdraw', 'quickdraw', 
    dataset_utils.get_quickdraw_dataset)

blurry_quickdraw_setting = graph_manager.Setting(
    'blurry_quickdraw', 'quickdraw', 'quickdraw', 'quickdraw', partial(
        dataset_utils.get_blurry_quickdraw_dataset, kern_len=5, kern_sig=0.5))

noisy_quickdraw_setting = graph_manager.Setting(
    'noisy_quickdraw', 'noisy_quickdraw', 'noisy_quickdraw', 'noisy_quickdraw', 
    dataset_utils.get_quickdraw_dataset)


############################################### ILSVRC
data_config={}
data_config['image_size'] = 256
data_config['crop_size'] = 224
data_config['image_dir'] = ILSVRC_BASE_PATH

# Source
base_ilsvrc_setting = graph_manager.Setting(
    'ilsvrc2012', 'ilsvrc2012_0-25', 'ilsvrc2012_30-50', 'ilsvrc2012_25-30',
    partial(dataset_utils.get_ilsvrc_dataset, **data_config))

# Target domains
hsv_ilsvrc_setting = graph_manager.Setting(
    'hsv_ilsvrc', 'ilsvrc2012_0-25', 'ilsvrc2012_30-50', 'ilsvrc2012_25-30', 
    partial(dataset_utils.get_hsv_ilsvrc_dataset, **data_config))

yuv_ilsvrc_setting = graph_manager.Setting(
    'yuv_ilsvrc', 'ilsvrc2012_0-25', 'ilsvrc2012_30-50', 'ilsvrc2012_25-30', 
    partial(dataset_utils.get_yuv_ilsvrc_dataset, **data_config))

fixed_scale_ilsvrc = graph_manager.Setting(
    'fixed_scale_ilsvrc', 'ilsvrc2012_0-25', 'ilsvrc2012_30-50', 'ilsvrc2012_25-30',
    partial(dataset_utils.get_fixed_downscaling_ilsvrc_dataset, **data_config, 
            scale=2., padding='EDGE'))

fixed_kaleido_ilsvrc = graph_manager.Setting(
    'fixed_kaleido_ilsvrc', 'ilsvrc2012_0-25', 'ilsvrc2012_30-50', 'ilsvrc2012_25-30',
    partial(dataset_utils.get_fixed_downscaling_ilsvrc_dataset, **data_config, 
            scale=1.8, padding='SYMMETRIC'))

fixed_rotation_ilsvrc = graph_manager.Setting(
    'fixed_rotation_ilsvrc', 'ilsvrc2012_0-25', 
    'ilsvrc2012_30-50', 'ilsvrc2012_25-30',
    partial(dataset_utils.get_fixed_rotation_ilsvrc_dataset, **data_config,
            angle=0.6))


##################################################################### PACS
data_config={}
data_config['image_size'] = 256
data_config['crop_size'] = None
data_config['image_dir'] = PACS_BASE_PATH

# Source
photo_pacs = graph_manager.Setting('photo', 'pacs_photo', 'pacs_photo', 'pacs_photo', 
                                   partial(dataset_utils.get_pacs_dataset, **data_config))

# Target domains
art_pacs = graph_manager.Setting('art', 'pacs_art_painting', 'pacs_art_painting', 'pacs_art_painting', 
                                 partial(dataset_utils.get_pacs_dataset, **data_config))

cartoon_pacs = graph_manager.Setting(
    'cartoon', 'pacs_cartoon', 'pacs_cartoon', 'pacs_cartoon', partial(dataset_utils.get_pacs_dataset, **data_config))

sketch_pacs = graph_manager.Setting('sketch', 'pacs_sketch', 'pacs_sketch', 'pacs_sketch',
                                    partial(dataset_utils.get_pacs_dataset, **data_config))


############################################ Main (create subsampled datasets)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate datasets')
    parser.add_argument('--mnist', action='store_true', help='Generate mnist related data')
    parser.add_argument('--cifar', action='store_true', help='Generate cifar/quickdraw rationed  data')
    parser.add_argument('--ilsvrc', action='store_true', help='Generate ilsvrc rationed  data')
    parser.add_argument('--pacs', action='store_true', help='Generate pacs rationed data')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    
    #### MNIST
    if args.mnist:
        # Split TFRecords       
        print(' > Split TFRecords') 
        preprocess_utils.split_tf_records('Data/mnist/tf_mnist_train', 'Data/mnist/tf_mnist0-25_train', 
                                         take=25000, shuffle=True, seed=args.seed, num_elements=55000)
        preprocess_utils.split_tf_records('Data/mnist/tf_mnist_train', 'Data/mnist/tf_mnist25-60_train',
                                         skip=25000, shuffle=True, seed=args.seed, num_elements=55000)
        preprocess_utils.split_tf_records('Data/mnist/tf_mnist-m_train', 'Data/mnist/tf_mnist-m25-60_train',
                                         skip=25000, shuffle=True, seed=args.seed, num_elements=55000)
        
        # Generate occlusion masks
        print('\n > Create occlusion masks')  
        for key in ['mnist25-60_train', 'mnist_val', 'mnist_test']:
            preprocess_utils.create_occlusion_masks('Data/mnist/tf_%s' % key, 'Data/mnist/tf_occluded_%s' % key, 
                                                    patch_size=0.5, seed=args.seed)
            
        # Generate transformation masks
        print('\n > Create transformations') 
        transform_config = {'min_angle': -1.5, 'max_angle': 1.5, 'min_scale': 1., 'max_scale': 2.5, 'seed': 42}
        for key in ['mnist25-60_train', 'mnist_val', 'mnist_test']:
            # Free transformation
            preprocess_utils.create_affine_transformation(
                'Data/mnist/tf_%s' % key, 'Data/mnist/tf_transformed_%s' % key, [14, 14], 
                **transform_config, seed=args.seed)
            # Fixed transformation
            preprocess_utils.create_affine_transformation(
                'Data/mnist/tf_%s' % key, 'Data/mnist/tf_fixed_transformed_%s' % key, [14, 14],
                fixed_transform=True, **transform_config, seed=args.seed)
        
        # Generate ratio subsampling
        print('\n > Generate ratio subsampling')  
        ratio_thresholds = [0.001, 0.01, 0.1, 1.]
        settings = [base_mnist_setting,
                    blurry_mnist_setting,
                    occluded_mnist_setting,
                    mnistm_setting, 
                    svhn_setting,
                    transformed_mnist_setting,
                    fixed_transformed_mnist_setting]
        setting_factory = graph_manager.SettingFactory(
        settings=settings,
        tfrecords_base_path='Data/mnist/tf_%s_%s',
        tfrecords_base_path_with_ratios='Data/mnist/ratios/tf_%s_%.4f_%s',
        ratio_thresholds=ratio_thresholds)
        setting_factory.create_subsampled_tfrecords(batch_size=50, seed=args.seed)
        
    
    #### CIFAR
    if args.cifar:  
        # Generate ratio subsampling
        print('\n > Generate ratio subsampling')  
        ratio_thresholds = [0.001, 0.01, 0.1, 1.]
        settings = [None,
                    cifar_setting,
                    noisy_cifar_setting,
                    quickdraw_setting,
                    noisy_quickdraw_setting]
        setting_factory = graph_manager.SettingFactory(
        settings=settings,
        tfrecords_base_path='Data/cifar/tf_%s_%s',
        tfrecords_base_path_with_ratios='Data/cifar/ratios/tf_%s_%.4f_%s',
        ratio_thresholds=ratio_thresholds)
        setting_factory.create_subsampled_tfrecords(batch_size=50, seed=args.seed)
        
        
    #### ILSVRC
    if args.ilsvrc:
        # Split TFRecords         
        print(' > Split TFRecords')
        preprocess_utils.split_tf_records('Data/ilsvrc2012/tf_ilsvrc2012_val', 'Data/ilsvrc2012/tf_ilsvrc2012_0-25_train', 
                                          take=25000, shuffle=True, seed=args.seed, num_elements=50000)
        preprocess_utils.split_tf_records('Data/ilsvrc2012/tf_ilsvrc2012_val', 'Data/ilsvrc2012/tf_ilsvrc2012_25-30_val', 
                                          skip=25000, take=5000, shuffle=True, seed=args.seed, num_elements=50000)
        preprocess_utils.split_tf_records('Data/ilsvrc2012/tf_ilsvrc2012_val', 'Data/ilsvrc2012/tf_ilsvrc2012_30-50_test', 
                                          skip=30000, shuffle=True, seed=args.seed, num_elements=50000)
        
        # Precompute occlusion masks      
        print('\n > Create occlusion masks')  
        for key in ['ilsvrc2012_0-25_train', 'ilsvrc2012_25-30_val', 'ilsvrc2012_30-50_test']:
            preprocess_utils.create_occlusion_masks(
                'Data/ilsvrc2012/tf_%s' % key, 'Data/ilsvrc2012/tf_occluded_%s' % key, patch_size=0.5, seed=args.seed)
            
        # Precompute transformation masks
        print('\n > Create transformations')  
        fixed_scale_config = {'min_angle': 0, 'max_angle': 0, 'min_scale': 1., 'max_scale': 2., 'seed': args.seed}
        fixed_rotation_config = {'min_angle': 0.8, 'max_angle': 0.8, 'min_scale': 0.85, 'max_scale': 0.85, 'seed': args.seed}
        for key in ['ilsvrc2012_0-25_train', 'ilsvrc2012_25-30_val', 'ilsvrc2012_30-50_test']:
            # fixed scaling
            preprocess_utils.create_affine_transformation(
                'Data/ilsvrc2012/tf_%s' % key, 'Data/ilsvrc2012/tf_fixed_scale_%s' % key, [112, 112], 
                fixed_transform=True, **fixed_scale_config)
            # fixed rotation
            preprocess_utils.create_affine_transformation(
                'Data/ilsvrc2012/tf_%s' % key, 'Data/ilsvrc2012/tf_fixed_rotation_%s' % key, [112, 112], 
                fixed_transform=True, **fixed_rotation_config)        
        
        # Generate ratio subsampling
        print('\n > Generate ratio subsampling')  
        ratio_thresholds = [0.01, 0.5, 1.]
        settings = [base_ilsvrc_setting,
                    blurry_ilsvrc,
                    noisy_ilsvrc_low,
                    noisy_ilsvrc_mid,
                    noisy_ilsvrc_high,
                    occluded_ilsvrc,
                    hsv_ilsvrc_setting,
                    yuv_ilsvrc_setting,
                    fixed_scale_ilsvrc, 
                    fixed_kaleido_ilsvrc,
                    fixed_rotation_ilsvrc]
        setting_factory = graph_manager.SettingFactory(
            settings=settings,
            tfrecords_base_path='./Data/ilsvrc2012/tf_%s_%s',
            tfrecords_base_path_with_ratios='./Data/ilsvrc2012/ratios/tf_%s_%.4f_%s' ,
            ratio_thresholds=ratio_thresholds)
        setting_factory.create_subsampled_tfrecords(batch_size=16, seed=42)
        
        
    #### PACS
    if args.pacs:
        # Generate ratio subsampling
        print('\n > Generate ratio subsampling')  
        ratio_thresholds = [0.01, 0.1, 1.]
        settings = [photo_pacs,
                    noisy_photo_pacs,
                    blurry_photo_pacs,
                    art_pacs,
                    cartoon_pacs,
                    sketch_pacs]
        setting_factory = graph_manager.SettingFactory(
            settings=settings,
            tfrecords_base_path='./Data/pacs/tf_%s_%s',
            tfrecords_base_path_with_ratios='./Data/pacs/ratios/tf_%s_%.4f_%s' ,
            ratio_thresholds=ratio_thresholds)
        setting_factory.create_subsampled_tfrecords(batch_size=16, seed=42)