import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import pickle

import settings
from include import classifiers
from include import graph_manager
from include import run_exp


"""Run as main.py [source_dataset] -g [num_gpus] -m [memory usage per gpu in percentage]

You also *have to* specify which experiments to run using --naive (apply 
source-pretrained network directly on the target domains)  or 
--ft (finetuning from different start point) or --ftflex (flextune 
different intermediate layers) or --all (all three)
"""


def backup_results(results, path):     
    """Pickle an object"""
    with open(path, 'wb') as f:
        pickle.dump(results, f)
        
        
def naive(config, classifier_constructor, verbose=True):
    results = config.copy()
    results['data'] = run_exp.run_pretrained(
        classifier_constructor,
        config['setting_factory'], 
        verbose=verbose, 
        config=config)
    backup_results(results, config['results_path'] % 'pretrained')

        
def finetune(config,
             layers, 
             classifier_constructor,
             save_model=True, 
             verbose=True):
    """Fine-tuning experiment"""
    results = config.copy()
    results['layers'] = layers
    results['data'] = run_exp.run_finetune_fromend(
        classifier_constructor,
        config['setting_factory'], 
        results['layers'],       
        save_model=[save_model] * len(results['layers']),
        log_file='txt_logs/finetune_%s.txt',
        exp_name='{base_name}/%s/ft%d/ratio%.2e'.format(base_name=config['base_name']),
        display_step=50 * int(verbose),
        base_config=config)
    backup_results(results, config['results_path'] % 'finetuning')
    
    
def flextune(config,
             layers, 
             classifier_constructor,
             save_model=True, 
             verbose=True):
    """Flex-tuning experiments"""
    results = config.copy()
    results['layers'] = layers
    results['data'] = run_exp.run_midtune_pointwise(
        classifier_constructor,
        config['setting_factory'],
        results['layers'],       
        save_model=[save_model] * len(results['layers']),
        log_file='txt_logs/flextune_%s.txt',
        exp_name='{base_name}/%s/ftflex%d/ratio%.2e'.format(base_name=config['base_name']),
        display_step=50 * int(verbose),
        base_config=config)
    backup_results(results, config['results_path'] % 'flextuning')
    
    
def run(exp_settings, 
        tfrecords_path,
        scratch_checkpoint_dir, 
        classifier_constructor, 
        config,
        run_naive=True,
        run_finetune=True,
        run_flextune=True,
        verbose=False):
    """Initialize and run the given set-up specified by a configuration dictionnary"""
    # Set-up configuration object
    config['results_path'] = 'results/{base_name}_results_%s.pkl'.format(base_name=config['base_name'])
    config['setting_factory'] = graph_manager.SettingFactory(
        settings=exp_settings,
        scratch_checkpoint_dir=scratch_checkpoint_dir,
        tfrecords_base_path='{tfrecords_path}/tf_%s_%s'.format(tfrecords_path=tfrecords_path),
        tfrecords_base_path_with_ratios='{tfrecords_path}/ratios/tf_%s_%.4f_%s'.format(tfrecords_path=tfrecords_path),
        ratio_thresholds=config['ratio_thresholds'],
        batch_size_per_ratio=config['batch_size_per_ratio'],
        num_epochs_per_ratio=config['num_epochs_per_ratio'],
        eval_step_per_ratio=config['eval_step_per_ratio'])
    backup_results(config, config['results_path'] % 'main')
    
    # Experiments
    if run_naive:
        print('\nNaive exp\n===================') 
        naive(config, classifier_constructor, verbose=verbose)
        
    if run_finetune:
        print('\nFinetuning exp\n===================')  
        # Finetune last, and 2 last layers + all layers
        layers = [1, 2, len(classifier_constructor.model_scopes) + 1]
        # For completeness: Finetune from all starting points
        # layers = list(range(2, len(classifier_constructor.model_scopes) + 1))
        finetune(config, layers, classifier_constructor, save_model=True, verbose=verbose)
    
    if run_flextune:
        print('\nFlextuning exp\n===================') 
        # Flextune all intermediate units one by one
        layers = list(range(1, len(classifier_constructor.model_scopes) + 1))
        flextune(config, layers, classifier_constructor, save_model=True, verbose=verbose)
        
        
def main():
    # Argparser
    parser = argparse.ArgumentParser(description='Flextuning experiments.')
    parser.add_argument('dataset', type=str, choices=[
        'mnist', 'pacs_from_imagenet'], help='Dataset')
    parser.add_argument('-g', '--gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('-m', '--mem_ratio', type=float, default=1.0, help='GPU memory fraction')
    parser.add_argument('--naive', action='store_true', help='pretrained experiments')
    parser.add_argument('--ft', action='store_true', help='ft< exp')
    parser.add_argument('--ftflex', action='store_true', help='ft. exp')
    parser.add_argument('--all', action='store_true', help='all exp')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
    args = parser.parse_args()
    if not (args.naive or args.ft or args.ftflex or args.all):
        return
    
    # Set up base configuration
    config = {}
    config['base_name'] = args.dataset
    config['num_train_devices'] = args.gpus
    config['gpu_memory_fraction'] = args.mem_ratio
    config['shuffle_buffer'] = 10000
    config['save_checkpoint_secs'] = 3600
    
    if config['base_name'] == 'mnist':
        # settings
        tfrecords_path = 'Data/mnist'
        exp_settings = [settings.base_mnist_setting,
                        settings.blurry_mnist_setting,
                        settings.occluded_mnist_setting,
                        settings.mnistm_setting, 
                        settings.svhn_setting]
        scratch_checkpoint_dir = "Data/pretrained_models/pretrained_mnist/"
        config['ratio_thresholds'] = [0.001, 0.01, 0.1, 1.]
        config['batch_size_per_ratio'] = [16, 32, 32, 32] # 1 epoch ~ ? steps = [1, 8, 78, 781]
        config['num_epochs_per_ratio'] = [1000, 500, 100, 25]
        config['eval_step_per_ratio'] = [20, 50, 500, 4000] # roughly every 5 epochs
        # Training hyperparameters        
        classifier_constructor = classifiers.MNISTClassifier
        config['image_size'] = 28
        config['learning_rate'] = 1e-4
        config['eval_epsilon'] = 0.0025
        config['min_train_steps'] = 250
        
    elif config['base_name'] == 'pacs_from_imagenet':
        tfrecords_path = 'Data/pacs'
        exp_settings = [settings.photo_pacs,
                        settings.art_pacs,
                        settings.cartoon_pacs,
                        settings.sketch_pacs]
        # settings
        config['ratio_thresholds'] = [0.01, 0.1, 1.]
        config['batch_size_per_ratio'] = [8 * args.gpus, 8 * args.gpus, 8 * args.gpus] 
        # 1 epoch ~ ? steps = [1, 16, 162] (1 GPUs)
        config['num_epochs_per_ratio'] = [2000, 1000, 150]  # 2000 - 2000 - 10000 steps
        config['eval_step_per_ratio'] = [100, 150, 800]       # every 5-10 epochs
        # Training hyperparameters  
        scratch_checkpoint_dir = "Data/pretrained_models/pretrained_ilsvrc2012/"
        classifier_constructor = classifiers.PACSFromImagenetClassifier
        config['image_size'] = 224
        config['crop_size'] = None
        config['learning_rate'] = 1e-4
        config['eval_epsilon'] = 0.025
        config['min_train_steps'] = 800
    else:
        raise NotImplementedError("Unknown base name", config['base_name'])
        
    # run experiments
    run(exp_settings,        
        tfrecords_path,
        scratch_checkpoint_dir,
        classifier_constructor,
        config,
        run_naive=args.naive or args.all,
        run_finetune=args.ft or args.all,
        run_flextune=args.ftflex or args.all,
        verbose=args.verbose)
    
if __name__ == '__main__':    
    main()