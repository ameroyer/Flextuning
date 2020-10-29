import sys
import numpy as np
from . import viz_utils

""" Utils function for running experiments. """

####################### Exp 1: Apply base pretrained model
def apply_pretrained(mode, 
                     classifier_constructor, 
                     setting, 
                     setting_factory,
                     verbose=False, 
                     config={}):
    """Apply the model pretrained on the base setting:
    
    Args:
        mode: Data split to apply on, one of 'viz_train', 'val' or 'test'
        classifier_constructor: graph_manager.classifier constructor
        setting: Setting to apply the model on
        setting_factor: A setting factory. Its base setting will be used to load the model checkpoint
        config: A configuration fed to the classifier constructor
    """
    assert mode in ['viz_train', 'val', 'test']
    if mode == 'viz_train':
        config['train_tfrecords'] = setting_factory.get_train_tf_records(setting)
    elif mode == 'val':
        config['val_tfrecords'] = setting_factory.get_val_tf_records(setting)
    elif mode == 'test':
        config['test_tfrecords'] = setting_factory.get_test_tf_records(setting)
    classifier = classifier_constructor(setting.get_dataset_fn, config=config)
    top1_accuracy, top5_accuracy = classifier.test(
        setting_factory.get_base_checkpoint_dir(), mode=mode, verbose=verbose)
    return top1_accuracy, top5_accuracy


def run_pretrained(classifier_constructor, setting_factory, verbose=False, config={}):
    """Apply pretrained base model on *all* settings of the given factory
    
    Returns array of size (num_settings, 2 (val/test), 2 (top-1 / top-5)))
    """
    results = np.zeros((setting_factory.get_num_settings() + 1, 2, 2))
    for i, setting in enumerate(setting_factory.get_all_settings()):
        print('\n\033[43m%s:\033[0m' % setting.name) 
        for m, mode in enumerate(['val', 'test']):
            top1_accuracy, top5_accuracy = apply_pretrained(
                mode, classifier_constructor, setting, setting_factory, verbose=verbose, config=config)
            results[i, m, 0] = top1_accuracy
            results[i, m, 1] = top5_accuracy
            if verbose:
                print('   > %s accuracy: top-1 = \033[31m%.5f\033[0m, top-5 = \033[31m%.5f\033[0m' % (
                    mode, top1_accuracy, top5_accuracy))
            else:
                print('   > %s accuracy: top-1 = %.5f, top-5 = %.5f' % (mode, top1_accuracy, top5_accuracy))
    return results


####################### Exp 2: Finetune and Midtune the model
def train_classifier(classifier_constructor,
                     dataset_fn,
                     trainable_scopes,
                     config,
                     display_step=1,
                     restore_model_scopes=None,
                     restore_model_dir=None,
                     save_checkpoint=False,
                     log_file=None):
    """ Encapsulate classifier training in a function."""
    # Train classifier
    classifier = classifier_constructor(dataset_fn, trainable_scopes=trainable_scopes, config=config)
    output = classifier.train_and_eval(
        save_checkpoint_secs=config.get('save_checkpoint_secs', 3600) if save_checkpoint else None,             
        restore_model_dir=restore_model_dir,
        restore_model_scopes=restore_model_scopes,
        display_step=display_step,
        verbose=False)
    # log 
    if log_file is not None:
        with open(log_file, 'a') as f:
            if save_checkpoint:
                f.write('\nsaved in %s' % classifier.save_log_dir)
            f.write('\n%s\n' % ('*' * 80))
            f.write(classifier.info)   
    return output
                
    
def finetune_training(classifier_constructor,
                      setting, 
                      setting_factory, 
                      list_finetuned_layers,
                      scratch_train=False,
                      save_model=None,
                      log_file=None,
                      exp_name='exp/%s/ft%d/ratio%.2e',
                      display_step=1,
                      base_config={}):  
    """ Fine-tuning experiments
    
    Args:
        classifier_constructor: graph_manager.classifier constructor
        model_scopes: List of string defining the trainable scopes in the model
        setting: The setting to train
        setting_factory: A Setting factory
        list_finetuned_layers: A list of list determining the layers to train in each experiment. 
             Each element is a list of integer that determines the trainable scopes
        save_models: Either None (no model is saved) or a list as long as list_finetuned_layers, indicating whether a model
            should be saved (with ratio 1.0)
        scratch_train: If True, performs additional experiments when training the model from scratch
        log_file: Optional log file
        exp_name: Base name for the log directory
        display_step: Frequency at which to display the training log
        base_config: Configuration dictionnary
        
    Returns:
        accuracies as a numpy array of size (number of ratios, number of layers setting, 2 (pretrained/ scratch), 
                                             2 (val / test), 2(top-1 / top-5)))
    """
    # Init the log file
    if log_file is not None:
        with open(log_file, 'w') as f:
            f.write('Finetuning %s, settings = %s %s' % (
                setting.name, list(setting_factory.get_ratios_settings()), list_finetuned_layers))
            f.write('\n%s' % ('*' * 120))
            
    # Config
    accuracies = np.zeros((setting_factory.get_num_ratios(), len(list_finetuned_layers), 2, 2, 2))
    config = base_config.copy()
    model_scopes = classifier_constructor.model_scopes
    config['val_tfrecords'] = setting_factory.get_val_tf_records(setting)
    config['test_tfrecords'] = setting_factory.get_test_tf_records(setting)
    save_model = [False] * len(list_finetuned_layers) if save_model is None else save_model
    assert len(save_model) == len(list_finetuned_layers)
    
    # For each ratio
    for i, (ratio, num_epochs, batch_size, eval_step) in enumerate(setting_factory.get_ratios_settings()):
        config['train_tfrecords'] = setting_factory.get_subsampled_train_tf_records(setting, ratio)
        config['num_epochs'] = num_epochs    
        config['batch_size'] = batch_size  
        config['eval_step'] = eval_step  
        
        # For each traianble layer settings
        for j, trainable_layers in enumerate(list_finetuned_layers):
            if len(trainable_layers) == 1 and trainable_layers[0] >= 0:
                config['exp_name'] = exp_name % (setting.name, trainable_layers[0], ratio)
            else:
                config['exp_name'] = exp_name % (setting.name, len(trainable_layers), ratio)
                
            # define scopes to train and to restore
            trainable_scopes = [model_scopes[x] for x in trainable_layers]
            restorable_scopes = [model_scopes]
           
            
            # If scratch_train, we also experiment with trainable layers initialized from scratch
            if scratch_train:
                frozen_scopes = set(model_scopes).difference(set(trainable_scopes))
                if len(frozen_scopes):
                    restorable_scopes.append(list(frozen_scopes))
        
            # Run expriments
            for k, restore_model_scopes in enumerate(restorable_scopes):
                # Update log file
                if log_file is not None:
                    with open(log_file, 'a') as f:
                        f.write('\n\n\nXP ratio %.4f, %d epochs, batch %d, %d num finetuned layer(s), %s' % (
                            ratio, num_epochs, batch_size, len(trainable_layers), 'pretrained' if k == 0 else 'scratch'))
                # Train
                save_checkpoint = (k == 0 and save_model[j])
                (num_train_steps, 
                 train_accuracy, 
                 val_top1_accuracy, 
                 val_top5_accuracy,
                 test_top1_accuracy, 
                 test_top5_accuracy) = train_classifier(
                    classifier_constructor,
                    setting.get_dataset_fn,
                    trainable_scopes,
                    config,
                    display_step=display_step,
                    restore_model_scopes=restore_model_scopes,
                    restore_model_dir=setting_factory.get_base_checkpoint_dir(),
                    save_checkpoint=save_checkpoint,
                    log_file=log_file)
                # Store
                accuracies[i, j, k, 0, 0] = val_top1_accuracy
                accuracies[i, j, k, 0, 1] = val_top5_accuracy
                accuracies[i, j, k, 1, 0] = test_top1_accuracy
                accuracies[i, j, k, 1, 1] = test_top5_accuracy                        
                # Display current state of the run_exp
                print('\r    ratio %.4f - %d layers (test=%.4f) (train=%.4f) (val=%.4f) (steps=%d) - %s%s' % (
                    ratio, len(trainable_layers), test_top1_accuracy, train_accuracy, val_top1_accuracy, 
                    num_train_steps, '(scratch)' if k > 0 else '(pre-trained)', ' ' * 40), end='')
                print()
                sys.stdout.flush()            
    return accuracies


def run_finetune_fromend(classifier_constructor,
                         setting_factory, 
                         num_finetuning_layers,
                         scratch_train=False,
                         save_model=None,
                         log_file=None,                         
                         exp_name='exp/%s/ftend%d/ratio%.2e',
                         display_step=1,
                         base_config={}):
    """Finetuning experiments from the end of the model (normal setting)
    
    Args:
        classifier_constructor: graph_manager.classifier constructor
        model_scopes: List of string defining the trainable scopes in the model
        setting_factory: A Setting factory
        num_finetuning_layers: number of layers to train (from the back)
        scratch_train: If True, performs additional experiments when training the model from scratch
        log_file: Optional log file
        exp_name: Base name for the log directory
        display_step: Frequency at which to display the training log
        base_config: Configuration dictionnary
        
    Returns:
        results, an array such that results[i, j, k, l] is the accuracy for training i, dataset ratio j, number of finetuned 
        layers k and pretrained_model (l = 0) or trained from scratch (l = 1)
    """
    # results storage
    results = np.zeros((setting_factory.get_num_settings(), 
                        setting_factory.get_num_ratios(), 
                        len(num_finetuning_layers), 
                        2, 2, 2))
    # trainable layers
    layers = [list(range(-1, - x - 1, -1)) for x in num_finetuning_layers]
    
    # run for each setting
    for i, setting in enumerate(setting_factory.variant_settings):
        print('\n\n\033[43m%s:\033[0m' % setting.name) 
        results[i, ...] = finetune_training(                                                    
            classifier_constructor,
            setting,                                                                   
            setting_factory,              
            layers, 
            scratch_train=scratch_train,
            save_model=save_model,
            log_file=log_file % setting.name if log_file is not None else log_file,
            exp_name=exp_name,
            display_step=display_step,
            base_config=base_config)
    sys.stdout.flush()
    return results
            
    
def run_midtune_pointwise(classifier_constructor,
                          setting_factory, 
                          num_finetuning_layers,
                          scratch_train=False,
                          save_model=None,
                          log_file=None,
                          exp_name='exp/%s/ftmid%d/ratio%.2e',
                          display_step=1,
                          base_config={}):
    """Midtuning experiments with only one trainable layer"""
    # results storage
    results = np.zeros((setting_factory.get_num_settings(), 
                        setting_factory.get_num_ratios(), 
                        len(num_finetuning_layers),
                        2, 2, 2))
    
    # trainable layers
    layers = [[x - 1] for x in num_finetuning_layers]
        
    # run for each setting
    for i, setting in enumerate(setting_factory.variant_settings):
        print('\n\n\033[43m%s:\033[0m' % setting.name)
        results[i, ...] = finetune_training(                                                    
            classifier_constructor, 
            setting,                                                                   
            setting_factory,              
            layers,
            scratch_train=scratch_train,
            save_model=save_model,
            log_file=log_file % setting.name if log_file is not None else log_file,
            exp_name=exp_name,
            display_step=display_step,
            base_config=base_config)
    sys.stdout.flush()
    return results


####################### Exp 3: Adapt the model (Section 5.4)
def gninut_training(classifier_constructor,
                    setting, 
                    setting_factory,
                    num_gninut_layers,
                    log_file=None,
                    exp_name='exp/%s/prep%d/ratio%.2e',
                    display_step=1,
                    base_config={}):  
    """ Gninut/Adapt experiments
    
    Args:
        classifier_constructor: graph_manager.classifier constructor
        setting_factory: A Setting factory
        num_gninut_layers: number of layers in the pix2pix model
        mult_num_epochs: Multiply the number of epochs by the given coefficient
        log_file: Optional log file
        exp_name: Base name for the log directory
        display_step: Frequency at which to display the training log
        base_config: Configuration dictionnary
        
    Returns:
        results, an array such that results[i, j, k, l, m] 
        is the accuracy for training i, dataset ratio j, number of layers k, val or test split (l) and top-1 or top-5 (m)
    """
    # Init log file
    if log_file is not None:
        with open(log_file, 'w') as f:
            f.write('Finetuning %s, settings = %s %s' % (
                setting.name, list(setting_factory.get_ratios_settings()), num_gninut_layers))
            f.write('\n%s' % ('*' * 120))
            
    # number of data ratios | number of fine-tuning layers
    accuracies = np.zeros((setting_factory.get_num_ratios(), len(num_gninut_layers), 2, 2))
    config = base_config.copy()
    config['val_tfrecords'] = setting_factory.get_val_tf_records(setting)    
    config['test_tfrecords'] = setting_factory.get_test_tf_records(setting)
    model_scopes = classifier_constructor.model_scopes
    
    # For each ratio
    for i, (ratio, num_epochs, batch_size, eval_step) in enumerate(setting_factory.get_ratios_settings()):
        config['train_tfrecords'] = setting_factory.get_subsampled_train_tf_records(setting, ratio)
        if 'mult_num_epochs' in config:
            config['num_epochs'] = int(num_epochs * config['mult_num_epochs'])  
        else:
            config['num_epochs'] = num_epochs
        config['batch_size'] = batch_size   
        config['eval_step'] = eval_step  
        
        # For each preprocessing mode
        for j, num_layers in enumerate(num_gninut_layers):
            config['num_preprocessing_layers'] = num_layers
            config['exp_name'] = exp_name % (setting.name, num_layers, ratio)
            trainable_scopes = ['.*preprocess/']
            
            # Allow multiple learning rates in case of different architectures
            if ('learning_rate' in base_config and 
                isinstance(base_config['learning_rate'], (list,)) and
                len(base_config['learning_rate']) == len(num_gninut_layers)):
                config['learning_rate'] = base_config['learning_rate'][j]                
            
            # Classifier
            classifier = classifier_constructor(setting.get_dataset_fn, trainable_scopes=trainable_scopes, config=config)
            (num_train_steps, train_accuracy, 
             val_top1_accuracy, val_top5_accuracy,
             test_top1_accuracy, test_top5_accuracy) = classifier.train_and_eval(
                save_checkpoint_secs=config.get('save_checkpoint_secs', None), 
                save_summaries_steps=config.get('save_summaries_steps', None),
                restore_model_dir=setting_factory.get_base_checkpoint_dir(),
                restore_model_scopes=model_scopes,
                display_step=display_step,
                verbose=False)            
            accuracies[i, j, 0, 0] = val_top1_accuracy
            accuracies[i, j, 0, 1] = val_top5_accuracy
            accuracies[i, j, 1, 0] = test_top1_accuracy
            accuracies[i, j, 1, 1] = test_top5_accuracy
            # Log
            if log_file is not None:
                with open(log_file, 'a') as f:
                    f.write('\n\n\nXP ratio %.4f, %d epochs, batch %d, %d preprocessing layer(s)' % (
                        ratio, num_epochs, batch_size, num_layers))
                    f.write('\nsaved in %s' % config['exp_name'])
                    f.write('\n%s\n' % ('*' * 80))
                    f.write(classifier.info)            
            # Display
            print('\r    ratio %.4f - %d layers (test=%.4f) (train=%.4f) (val=%.4f) (steps=%d)%s' % (
                ratio, num_layers, test_top1_accuracy, train_accuracy, val_top1_accuracy, num_train_steps, ' ' * 40), end='')
            print()            
    return accuracies


def run_gninut(classifier_constructor,
               setting_factory, 
               num_gninut_layers,
               log_file=None,
               display_step=1,
               exp_name='exp/%s/prep%d/ratio%.2e',
               base_config={}):
    """Midtuning experiments with only one trainable layer"""
    results = np.zeros((setting_factory.get_num_settings(), 
                        setting_factory.get_num_ratios(), 
                        len(num_gninut_layers),
                        2, 2))
    for i, setting in enumerate(setting_factory.variant_settings):
        print('\n\n\033[43m%s:\033[0m' % setting.name) 
        results[i, ...] = gninut_training(classifier_constructor,
                                           setting, 
                                           setting_factory,
                                           num_gninut_layers,
                                           log_file=log_file % setting.name,
                                           exp_name=exp_name,
                                           display_step=display_step,
                                           base_config=base_config)
    sys.stdout.flush()
    return results