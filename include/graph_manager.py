import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
from datetime import datetime
from functools import partial
from collections import defaultdict

import numpy as np
import tensorflow as tf

from . import dataset_utils
from . import net_utils
from . import tfrecords_utils
from . import viz_utils


class Setting(object):
    """Defines a Setting object."""
    def __init__(self, 
                 name,
                 train_tf_records,
                 test_tf_records,
                 val_tf_records,
                 get_dataset_fn):
        """A Setting object is composed of:
            name: A string name
            train_tf_records: Train tf records base name
            test_tf_records: Test tfrecords base name
            val_tf_records: Val tfrecords base name (optional)
            get_dataset_fn: A function from dataset_utils
        """
        self.name = name
        self.train_tf_records = train_tf_records
        self.test_tf_records = test_tf_records
        self.val_tf_records = val_tf_records
        self.get_dataset_fn = get_dataset_fn
        
        
class SettingFactory(object):
    """Manages all settings."""
    def __init__(self, 
                 settings,
                 tfrecords_base_path,
                 tfrecords_base_path_with_ratios, 
                 scratch_checkpoint_dir='./log_%s',
                 ratio_thresholds=[1.0],
                 num_epochs_per_ratio=None,
                 batch_size_per_ratio=None,
                 eval_step_per_ratio=None):
        """A SettingFactory is composed of:
            settings: list of setting objects. The  first item will be used as base setting
            scratch_checkpoint_dir: Format string for checkpoint directory
            tfrecords_base_path: Format string for tfrecords 
            tfrecords_base_path_with_ratios: Format string for subsampled tfrecords
            ratio_thresholds: ratio thresholds to consider in experiments
            num_epochs_per_ratio: Optional list to use different number of epochs for each ratio
            batch_size_per_ratio: Optional list to use different batch size for each ratio   
            eval_step_per_ratio: Optional list to determine the early stopping criterion check frequency
        """
        self.base_setting = settings[0]
        self.variant_settings = settings[1:]
        self.scratch_checkpoint_dir = scratch_checkpoint_dir
        self.tfrecords_base_path = tfrecords_base_path
        self.tfrecords_base_path_with_ratios = tfrecords_base_path_with_ratios
        self.ratio_thresholds = ratio_thresholds
        # optional number of epochs
        num_ratios = len(self.ratio_thresholds)
        self.num_epochs_per_ratio = num_epochs_per_ratio
        if self.num_epochs_per_ratio is None:
            self.num_epochs_per_ratio = [1] * num_ratios
        assert len(self.num_epochs_per_ratio) == num_ratios
        # optional batch size
        self.batch_size_per_ratio = batch_size_per_ratio
        if self.batch_size_per_ratio is None:
            self.batch_size_per_ratio = [1] * num_ratios
        assert len(self.batch_size_per_ratio) == num_ratios
        # optional eval step
        self.eval_step_per_ratio = eval_step_per_ratio
        if self.eval_step_per_ratio is None:
            self.eval_step_per_ratio = [-1] * num_ratios
        assert len(self.eval_step_per_ratio) == num_ratios
                
    def get_ratios_settings(self):
        """Returns a list of tuples (threshold, num epochs, batch_size, validation step) for each ratio"""
        return zip(self.ratio_thresholds, 
                   self.num_epochs_per_ratio, 
                   self.batch_size_per_ratio,
                   self.eval_step_per_ratio)
    
    def get_num_settings(self):
        """Returns the number of variant settings"""
        return len(self.variant_settings)
    
    def get_num_ratios(self):
        """Returns the number of data ratios"""
        return len(self.ratio_thresholds)       
    
    def get_settings(self):
        """Returns all the variant settings"""
        return self.variant_settings 
    
    def get_all_settings(self):
        """Returns *all* settings, including the base one"""
        return [self.base_setting] + self.variant_settings
    
    def get_all_settings_names(self):
        """Returns all settings names"""
        return [x.name for x in self.get_all_settings()]
        
    def get_train_tf_records(self, setting):
        """Returns train tfrecords for the given setting."""
        return self.tfrecords_base_path % (setting.train_tf_records, 'train')
    
    def get_subsampled_train_tf_records(self, setting, ratio):
        """Returns subsampled train tfrecords for the given setting and ratio."""
        return self.tfrecords_base_path_with_ratios % (setting.train_tf_records, ratio, 'train')
        
    def get_test_tf_records(self, setting):
        """Returns test tfrecords for the given setting."""
        return self.tfrecords_base_path % (setting.test_tf_records, 'test')
    
    def get_val_tf_records(self, setting):
        """Returns val tfrecords for the given setting."""
        return None if setting.val_tf_records is None else self.tfrecords_base_path % (setting.val_tf_records, 'val')
    
    def get_base_checkpoint_dir(self):
        """Returns checkpoint directory for the pretrained base model."""
        return self.scratch_checkpoint_dir #% self.base_setting.name
        
    def display_settings(self, modes=['train', 'val', 'test'], num_displays=8, 
                         figwidth=16, base_config={'batch_size': 8}, verbose=False):
        """Vizualize dataset samples for each setting.
        
        Args:
            modes: Split to visualize. Must be a subset of ['train', 'val', 'test']
            num_displays: Number of samples to display
            figwidth: Figure width
            config: Additional configuration for the dataset
            verbose: If True, browse the whole dataset and prints the count and class repartition of the samples
        """
        config = base_config.copy()
        for i, setting in enumerate([self.base_setting] + self.variant_settings):    
            print('\033[043m[Base] %s\033[0m' % setting.name if i == 0 else '\033[043m%s\033[0m' % setting.name)
            for mode in modes:
                print('   > %s' % mode)
                config['%s_tfrecords' % mode] = getattr(self, 'get_%s_tf_records' % mode)(setting)
                viz_utils.display_dataset(setting.get_dataset_fn, mode=mode, num_displays=num_displays, 
                                          figwidth=figwidth, config=config, verbose=verbose)
            print()                
            
    def create_subsampled_tfrecords(self, batch_size=32, seed=42):
        """Create tfrecords with subsets of the data to train in different settings.
        
        Args:
            batch_size: The batch size to use to load the dataset when computing class ratios
            seed: Fix the random seed
        """
        seen = {}
        for setting in self.variant_settings:   
            if not setting.train_tf_records in seen:
                # Precompute class ratios
                print('\033[043m%s\033[0m (%s)' % (setting.name, setting.train_tf_records))
                seen[setting.train_tf_records] = True
                config = {'train_tfrecords': self.get_train_tf_records(setting), 'batch_size': batch_size}
                class_ratios = tfrecords_utils.get_class_ratios(dataset_utils.get_class_only_dataset, config, seed=seed)
                # SaveTFRecords
                for ratio in self.ratio_thresholds:
                    src_path = self.get_train_tf_records(setting)
                    tgt_path = self.get_subsampled_train_tf_records(setting, ratio)
                    dir_name = os.path.dirname(os.path.realpath(tgt_path))
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name)
                    writer = tf.python_io.TFRecordWriter(tgt_path)
                    for cr, example in zip(class_ratios, tf.python_io.tf_record_iterator(src_path)):
                        if cr < ratio: writer.write(example)
                    writer.close()      
                    print('   Wrote in \033[036m%s\033[0m' % tgt_path)
                    print()
            
        
class Classifier(object):
    """Defines a Classifier object."""
    def __init__(self, 
                 get_dataset_fn, 
                 feed_forward_fn, 
                 preprocess_fn=None, 
                 transition_fn=None, 
                 trainable_scopes=None,
                 scope_name='net',
                 config={}):
        """
        Args:
            get_dataset_fn: has signature
                * mode, one of 'train' or 'test'
            feed_forward_fn: has signature
                * images, a 4D Tensor
                * is_training
                * verbose, the verbosity level
                Main function that applies the classification 
            preprocess_fn: has signature
                * images
                * is_training
                Preprocess function image to image
            transition_fn takes image as inputs and returns images. This function is
                applied between preprocess fn and feed_fotward_fn
            trainable_scopes: Specifies a subset of scopes to train. If None, trains all trainable_variales.
            scope_name: Main scope name
            config: Main configuration directory fed as kwargs to the *_fn functions.
        """  
        ## Graph config
        self.get_dataset_fn = get_dataset_fn
        self.preprocess_fn = preprocess_fn
        self.transition_fn = transition_fn
        self.feed_forward_fn = feed_forward_fn  
        
        self.scope_name = scope_name    
        self.preprocess_scope_name = '%s_preprocess' % self.scope_name
        self.trainable_scopes = trainable_scopes 
        self.config = config
        
        ## Optional config
        # Number of (GPU) training devices    (default to 1)
        # Memory fraction to use per device   (default to 1.)
        # Base learning rate                  (default to 1e-3)
        # Early stopping criterion steps      (default to -1)
        # Minimum number of training steps    (default to 1)
        self.num_devices = config.get('num_train_devices', 1)
        self.gpu_memory_fraction = config.get('gpu_memory_fraction', 1.)
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.eval_step = config.get('eval_step', -1)
        self.eval_epsilon = config.get('eval_epsilon', 0.001)
        self.min_steps = config.get('min_train_steps', 1)
        
        ## Info string
        self.write_info = False
        self.info = ('Config\n------\n' + 
                     '\n'.join('*%s* = %s' % (k, self.config[k]) for k in sorted(self.config.keys())))
        self.info += '\n*dataset_fn* = %s' % self.get_dataset_fn
        self.info += '\n*preprocess_fn* = %s' % self.preprocess_fn
        self.info += '\n*transition_fn* = %s' % self.transition_fn
        self.info += '\n*feed_forward_fn* = %s' % self.feed_forward_fn
        
        ## Init graph
        self.graph = tf.Graph()
        self.save_log_dir = None
        with self.graph.as_default():
            self.trainable_variables = None 
            self.loss = None
            self.train_op = None
            self.train_summary_op = None
            self.stop_summary_hook = lambda: None
            self.train_accuracy = None
            self.last_val_accuracy = 0.
            self.val_accuracy = None
            self.val_top5_accuracy = None
            self.num_val_samples = None
            self.test_accuracy = None
            self.test_top5_accuracy = None
            self.num_test_samples = None
            
    def reset_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.trainable_variables = None 
            self.loss = None
            self.train_op = None
            self.train_summary_op = None
            self.stop_summary_hook = lambda: None
            self.train_accuracy = None
            self.last_val_accuracy = 0.
            self.val_accuracy = None
            self.val_top5_accuracy = None
            self.num_val_samples = None
            self.test_accuracy = None
            self.test_top5_accuracy = None
            self.num_test_samples = None
            
    def feed_forward(self, inputs, config, is_training=True, reuse=False, is_chief=True, verbose=False):
        """A feed forward pass to get logits from inputs"""
        assert 'image' in inputs
        net = inputs['image']
        # Preprocess
        with tf.variable_scope(self.preprocess_scope_name, reuse=reuse):
            if self.preprocess_fn is not None:
                net = self.preprocess_fn(net, is_training=is_training, reuse=reuse, verbose=verbose, 
                                         is_chief=is_chief, **config)
            if self.transition_fn is not None:
                net = self.transition_fn(net, is_chief=is_chief)
        ### Feed-forward
        with tf.variable_scope(self.scope_name, reuse=reuse):
            logits = self.feed_forward_fn(net, is_training=is_training, reuse=reuse, verbose=verbose, 
                                          is_chief=is_chief, **config) 
        return logits
        
    def train_pass(self, reuse=False, verbose=True):
        """Train pass"""        
        
        if verbose: print('   Using \033[31m%d\033[0m train devices' % self.num_devices)
        self.info += '\n   Using *%s* train devices' % self.num_devices
        
        dev_config = self.config.copy()
        dev_config['batch_size'] = dev_config['batch_size'] // self.num_devices
        self.info += '\n   Using *%d* batch size per device' % dev_config['batch_size']
        
        for dev in range(self.num_devices):
            with tf.device('/gpu:%d' % dev): 
                ### Config
                is_chief = (dev == 0)
                dev_verbose = verbose and is_chief   
                dev_reuse = reuse or (dev > 0)  
                ### Feed 
                inputs = self.get_dataset_fn('train', shards=self.num_devices, index=dev, **dev_config).get_next()
                logits = self.feed_forward(
                    inputs, dev_config, is_training=True, reuse=dev_reuse, is_chief=is_chief, verbose=dev_verbose)
                ### Loss and accuracy
                dev_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inputs['class'], logits=logits)
                dev_loss = tf.reduce_mean(dev_loss)
                tf.add_to_collection('classification_loss', dev_loss)                
                preds = tf.argmax(logits, axis=1, output_type=tf.int32)
                accuracy = tf.reduce_mean(tf.to_float(tf.equal(inputs['class'], preds)))
                tf.add_to_collection('classification_accuracy', accuracy)                
                
        ## Collect accross devices
        self.loss = tf.reduce_mean(tf.stack(tf.get_collection('classification_loss'), axis=0), axis=0)
        tf.summary.scalar('classification_loss', self.loss, collections=['train'])
        self.train_accuracy = tf.reduce_mean(tf.stack(tf.get_collection('classification_accuracy'), axis=0), axis=0)
        tf.summary.scalar('classification_accuracy', self.train_accuracy, collections=['train'])
        
        ## Determine variables to train
        if self.trainable_scopes is None:
            self.trainable_variables = tf.trainable_variables()
        else:
            self.trainable_variables = [v for scope in self.trainable_scopes for v in tf.trainable_variables(scope=scope)]
        self.info += ('\n\nTrainable variables\n-----------------\n' + 
                      '\n'.join('    %s' % v.name for v in self.trainable_variables))
        
        ## Create train operation
        if verbose: 
            print('   Train with Adam Optimizer and lr = \033[36m%.4f\033[0m:' % self.learning_rate)
            print('\n'.join('      > %s' % v.name for v in self.trainable_variables))
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.global_step = tf.train.get_or_create_global_step()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if verbose: print('   ', len(update_ops), 'update operations found')
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step, var_list=self.trainable_variables)  
            
        ## Add summaries        
        train_summaries = tf.get_collection('train')
        self.train_summary_op = tf.summary.merge(train_summaries) if len(train_summaries) else None

    def eval_pass(self, mode, shuffle_test=False, reuse=True, verbose=False):
        """Evaluation pass
        
        Args:
            mode: one of 'val', 'test' or 'viz_train' (one-pass iterators)
            shuffle_test: Only used if mode is `test`. Shuffle the dataset iterator with the shuffle_buufer in self.config.
            reuse: reuse variables in the model
            verbose: verbosity level
            
        Returns:
        """
        # inputs
        assert mode in ['val', 'test', 'viz_train']
        iterator = self.get_dataset_fn(mode, shuffle_test=shuffle_test, **self.config)
        tf.add_to_collection('iterator_initializer', iterator.initializer)
        inputs = iterator.get_next() 
        # feed-forward
        logits = self.feed_forward(inputs, self.config, is_training=False, reuse=reuse, verbose=verbose) 
        num_samples = tf.to_int32(tf.shape(logits)[0])
        labels = inputs['class']
        # top-1 accuracy (cumulative)
        preds = tf.argmax(logits, axis=1, output_type=tf.int32)
        accuracy = tf.reduce_sum(tf.to_float(tf.equal(labels, preds)))
        # top-5 accuracy (cumulative)
        k = min(5, logits.get_shape()[-1])
        top5_preds = tf.to_int32(tf.nn.top_k(logits, k=k)[1])
        top5_labels = tf.tile(tf.expand_dims(labels, axis=-1), (1, k))
        top5_accuracy = tf.reduce_sum(tf.to_float(
            tf.reduce_any(tf.equal(top5_preds, top5_labels), axis=-1)))
        # return
        return iterator, labels, preds, accuracy, top5_accuracy, num_samples        
        
    def test_pass(self, shuffle_test=False, reuse=True, verbose=False):
        """Test set pass"""
        # Add tensors to compute metrics
        _, labels, preds, accuracy, top5_accuracy, num_samples = self.eval_pass(
            'test', shuffle_test=shuffle_test, reuse=reuse, verbose=verbose)
        self.test_accuracy = accuracy
        self.test_top5_accuracy = top5_accuracy
        self.num_test_samples = num_samples
        # Additional optional outputs for viz_results_utils.gninut_vizualize_filtered
        return tf.equal(labels, preds), labels, preds
        
    def validation_pass(self, reuse=True, verbose=False):
        """Validation set pass"""
        # Add tensors to compute metrics
        iterator, _, _, accuracy, top5_accuracy, num_samples = self.eval_pass('val', reuse=reuse, verbose=verbose)
        self.val_iterator = iterator
        self.val_accuracy = accuracy
        self.val_top5_accuracy = top5_accuracy
        self.num_val_samples = num_samples
        
    def get_session(self, 
                    mode='train',
                    save_log_dir=None, 
                    save_summaries_steps=None,
                    save_checkpoint_secs=None,
                    restore_model_dir=None,
                    restore_model_scopes=None, 
                    exclude_restore_model_scopes=None, 
                    replace_restore_model_scopes=None,
                    verbose=True):
        """Get a standard monitored session object.
        
        Args:
            mode: train (add checkpoint and summary saver) or test (basic session)
            save_log_dir: Log directory
            save_summaries_steps: frequency to save train summaries
            save_checkpoint_secs: frequency to save train checkpoint
            restore_model_dir: directory to restore checkpoint from. No restoration if not given.
            restore_model_scopes: which scopes to restore from the checkpoints. Defaults to all variables if not given.
            exclude_restore_model_scopes: Optional scope names to exclude while restoring.
            replace_restore_model_scopes: If given, strip this header from all restore_model_scopes
        """   
        ### Restore model
        init_fn = lambda scaffold, sess: None
        if restore_model_dir is not None and (restore_model_scopes is None or len(restore_model_scopes)):
            model_path = tf.train.latest_checkpoint(restore_model_dir)
            assert model_path is not None
            if verbose: print('   Loading %s checkpoint from \033[36m%s\033[0m' % (self.scope_name, model_path))    
            self.info += '\n   Loading %s checkpoint from \033[36m%s\033[0m' % (self.scope_name, model_path)
            # get variables to restore (exclude ADAM)
            restore_variables = tf.contrib.framework.get_variables_to_restore(
                include=restore_model_scopes, exclude=exclude_restore_model_scopes)
            restore_variables = [v for v in restore_variables if not v.name.rsplit('/', 1)[-1].startswith('Adam')]
            self.info += '\n   Restoring variables\n'
            self.info += '\n'.join('      > %s' % v.name for v in restore_variables)
            # replace variable op name if needed
            if replace_restore_model_scopes is not None:
                for (src, tgt) in replace_restore_model_scopes:
                    restore_variables = {v.op.name.replace(src, tgt): v for v in restore_variables}        
            # assignment function
            assign_fn = tf.contrib.framework.assign_from_checkpoint_fn(
                model_path, restore_variables, ignore_missing_vars=True)
            init_fn = lambda scaffold, sess, assign_fn=assign_fn: assign_fn(sess)
         
        ### Log dir
        self.save_log_dir = save_log_dir
        if self.save_log_dir is None:
            exp_name = self.config['exp_name'] if 'exp_name' in self.config else 'exp'
            self.save_log_dir = os.path.join('./log', exp_name, datetime.now().strftime("%m-%d_%H-%M"))
        
        ### Scaffold (initializers)
        init_iterator_op = tf.get_collection('iterator_initializer')
        local_init_op = tf.group(tf.local_variables_initializer(), *init_iterator_op)
        scaffold = tf.train.Scaffold(init_fn=init_fn, local_init_op=local_init_op,
                                     ready_op=tf.constant([]), ready_for_local_init_op=tf.constant([])) 
        
        ### Other configs
        if self.gpu_memory_fraction < 1.0:
            config = tf.ConfigProto(
                allow_soft_placement=True, 
                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction))
        else:
            config = tf.ConfigProto(allow_soft_placement=True)
        session_creator = tf.train.ChiefSessionCreator(scaffold=scaffold, checkpoint_dir=None, config=config)
        
        ### Train Hooks (regularly save summaries and checkpoints)
        hooks = []
        if mode =='train':
            # image summary hook
            if self.train_summary_op is not None and save_summaries_steps is not None:
                self.write_info = True
                summaryhook = tf.train.SummarySaverHook(save_steps=save_summaries_steps,
                                                        output_dir=self.save_log_dir,
                                                        summary_op=self.train_summary_op)
                hooks.append(summaryhook)
                def stop_summary_hook():
                    summaryhook._next_step = -1
                self.stop_summary_hook = stop_summary_hook
            # checkpoint saver hook during training
            if save_checkpoint_secs is not None:
                if verbose: print('   Saving checkpoint in \033[36m%s\033[0m' % self.save_log_dir)
                self.info += '\n   Saving checkpoint in \033[36m%s\033[0m' % self.save_log_dir
                saver = tf.train.Saver(max_to_keep=1)
                hooks.append(tf.train.CheckpointSaverHook(self.save_log_dir, saver=saver, save_secs=save_checkpoint_secs))
                
        ### Return session
        return tf.train.MonitoredSession(session_creator=session_creator, hooks=hooks)            
        
    def train_and_eval(self,
                       display_step=1,
                       write_step=50,
                       save_checkpoint_secs=None,
                       restore_model_dir=None,
                       restore_model_scopes=None,
                       exclude_restore_model_scopes=None, 
                       replace_restore_model_scopes=None,
                       save_log_dir=None,
                       save_summaries_steps=None,
                       verbose=True):
        """Train and evaluate the model.
        
        Args:
            epsilon: Stop training if the validation accuracy is above the latest one with an epsilon difference
            display_step: Display loss frequency every given step
            write_step: At which frequency to write in the info string
            save_checkpoint_secs: Whether to save checkpoints and at which interval
            restore_model_dir: Directory to restore checkpoint from (otherwise None)
            restore_model_scopes: If given, only restore variables under the given scopes
            exclude_restore_model_scopes: Optional scope names to exclude while restoring.
            replace_restore_model_scopes: If given, strip this header from all restore_model_scopes
            save_log_dir: Directory to save log outputs to if any (if None, automatically define)
            save_summaries_steps: Frequency to save train summaries
            verbose: verbose output
        """
        with self.graph.as_default():
            ### Build the graph
            if verbose: print('\n\033[43mGraph:\033[0m')
            self.train_pass(reuse=False, verbose=verbose)
            self.test_pass(reuse=True, verbose=False)  
            if self.eval_step > 0:
                self.validation_pass(reuse=True, verbose=False)
            
            ### Info File
            if verbose: print('\n\033[43mTrain:\033[0m')
            self.info += '\n\nTrain log\n--------'
            self.info += '\n   learning rate = %.4f' % self.learning_rate
            self.info += '\n   %d epochs (check early stop step = %d with epsilon = %.5f)' % (
                self.config['num_epochs'], self.eval_step, self.eval_epsilon)
            
            ### Session
            with self.get_session('train',
                                  save_log_dir=save_log_dir,
                                  save_summaries_steps=save_summaries_steps,
                                  save_checkpoint_secs=save_checkpoint_secs,
                                  restore_model_dir=restore_model_dir,
                                  restore_model_scopes=restore_model_scopes, 
                                  exclude_restore_model_scopes=exclude_restore_model_scopes, 
                                  replace_restore_model_scopes=replace_restore_model_scopes,
                                  verbose=verbose) as sess:
                try:
                    # Init validation accuracy on the loaded pre-trained model
                    if self.eval_step > 0:
                        sess.run(self.val_iterator.initializer)
                        self.last_val_accuracy = eval_accuracy(
                            sess, [self.val_accuracy], self.num_val_samples, verbose=verbose)[0][0]
                        self.info += '\n Pretrained model accuracy on val = %.4f' % self.last_val_accuracy
                        # Test
                        self.last_val_accuracy = 0
                        
                    # Train
                    while 1:                        
                        # Update parameter
                        global_step, train_loss, train_accuracy, _ = sess.run([
                            self.global_step, self.loss, self.train_accuracy, self.train_op])
                        if display_step > 0 and global_step % display_step == 0: 
                            print(('\r   \033[33m(train)\033[0m step %08d: ' % global_step +
                                   'loss = \033[31m%.4f\033[0m, ' % train_loss +
                                   'train_acc = \033[34m%.4f\033[0m, ' % train_accuracy +
                                   'val_acc = \033[34m%.4f\033[0m' % self.last_val_accuracy), end='')
                            
                        # Info log
                        if global_step % write_step == 0:
                            self.info += '\n   step %d: train_loss = %.4f, train_batch_acc = %.4f, val_acc = %.4f' % (
                                global_step, train_loss, train_accuracy, self.last_val_accuracy)
                            
                        # Early stopping 
                        if self.eval_step > 0 and global_step > self.min_steps and global_step % self.eval_step == 0:
                            sess.run(self.val_iterator.initializer)
                            val_acc = eval_accuracy(sess, [self.val_accuracy], self.num_val_samples, verbose=False)[0][0]
                            if self.last_val_accuracy - val_acc > self.eval_epsilon:
                                self.info += '\n   Early stopping at step %d (val_acc = %.3f, last = %.3f)' % (
                                    global_step, val_acc, self.last_val_accuracy)
                                self.last_val_accuracy = val_acc
                                break
                            self.last_val_accuracy = val_acc
                except tf.errors.OutOfRangeError:
                    pass
                except KeyboardInterrupt:
                    print('Keyboard interrupted')   
                self.info += '\n   final step %d: train_loss = %.4f, train_batch_acc = %.4f, val_acc = %.4f' % (
                    global_step, train_loss, train_accuracy, self.last_val_accuracy)  
                self.stop_summary_hook() # Hack: Stop the training saver hooks to feed off the train iterator
                
                if verbose: print('\n\033[43mEval:\033[0m')
                # Eval on validation set
                if self.eval_step > 0:
                    sess.run(self.val_iterator.initializer)
                    [val_top1_accuracy, val_top5_accuracy], num_val_samples = eval_accuracy(
                        sess, [self.val_accuracy, self.val_top5_accuracy], self.num_val_samples, verbose=verbose) 
                    self.info += '\n\n   val accuracy = %.4f, top5 = %.4f, (%d val samples)' % (
                        val_top1_accuracy, val_top5_accuracy, num_val_samples)
                else:
                    val_top1_accuracy, val_top5_accuracy = 0., 0.
                
                # Eval on test set
                [test_top1_accuracy, test_top5_accuracy], num_test_samples = eval_accuracy(
                    sess, [self.test_accuracy, self.test_top5_accuracy], self.num_test_samples, verbose=verbose) 
                self.info += '\n\n   test accuracy = %.4f, top5 = %.4f, (%d test samples)' % (
                    test_top1_accuracy, test_top5_accuracy, num_test_samples)
                
                # Return
                return (global_step, 
                        train_accuracy, 
                        val_top1_accuracy, 
                        val_top5_accuracy,
                        test_top1_accuracy,
                        test_top5_accuracy)

    def test(self,
             restore_model_dir, 
             mode='test',
             restore_model_scopes=None,
             exclude_restore_model_scopes=None, 
             replace_restore_model_scopes=None,
             verbose=True):
        """Load a checkpoint and evaluate the model.
        
        Args:
            restore_log_dir: Directory to restore checkpoint from
            split: Split of the dataset to evaluate on. One of ['viz_train', 'val', 'test']
            restore_model_scopes: Which scopes to restore, defaults to all 
            exclude_restore_model_scopes: Optional scope names to exclude while restoring.
            replace_restore_model_scopes: If given, strip this header from all restore_model_scopes
            verbose: verbose output
        """
        assert mode in ['viz_train', 'val', 'test']
        with self.graph.as_default():
            _, _, _, top1_accuracy, top5_accuracy, num_samples = self.eval_pass(mode, reuse=False, verbose=False)
            self.info += '\n\n%s evaluation log\n--------' % mode
            with self.get_session('test', 
                                  restore_model_dir=restore_model_dir, 
                                  restore_model_scopes=restore_model_scopes,
                                  exclude_restore_model_scopes=exclude_restore_model_scopes, 
                                  replace_restore_model_scopes=replace_restore_model_scopes,
                                  verbose=verbose) as sess:
                [top1_accuracy_, top5_accuracy_], num_samples_ = eval_accuracy(
                    sess, [top1_accuracy, top5_accuracy], num_samples, verbose=verbose) 
                self.info += '\n\n   %s accuracy = %.4f, top5 = %.4f, (%d %s samples)' % (
                    mode, top1_accuracy_, top5_accuracy_, num_samples_, mode)   
                return top1_accuracy_, top5_accuracy_
            
    def display_tensors(self, 
                        tensors_names, 
                        restore_model_dir, 
                        restore_model_scopes=None,
                        figwidth=5, 
                        shuffle_test=False,
                        verbose=False):
        """Load a checkpoint and display the relevant tensors (weights or activations).
        
        Args:
            tensors_names: A list of pairs (tensor_name, tensor_type). Where tensor_type is one of 
                `weights` or `activations` and is used to choose the correct display method
            restore_model_dir: Directory to restore checkpoint from
            restore_model_scopes: Which scopes to restore, defaults to all
            figwidth: The width of the matplotlib figure
            shuffle_test: Whether to shuffle the test set to display varied samples
            verbose: verbose output
        """
        with self.graph.as_default():
            self.test_pass(shuffle_test=shuffle_test, reuse=False, verbose=verbose)
            relevant_tensors = {}
            
            for name, display_type in tensors_names:
                t = self.graph.get_tensor_by_name(name)
                dims = t.get_shape().as_list()
                grid_dims = viz_utils.get_grid_size(dims[-1])
                # Weights have shape (w, h, c_in, c_out)
                # We normalize them over (w, h)
                # We make them into a grid first based on c_out, then based on c_in
                # Final tensor has shape (1, ?, ?, 1) (or 3 if c_in == 3, eg first conv layer)
                if display_type == 'weights':
                    # shape and normalize
                    t = tf.transpose(t, (3, 0, 1, 2))
                    t -= tf.reduce_min(t, axis=(1, 2), keep_dims=True)
                    t /= tf.reduce_max(t, axis=(1, 2), keep_dims=True) + 1e-8
                    # grid
                    if dims[2] == 3 or dims[2] == 1:
                        relevant_tensors[name] = viz_utils.image_grid(t, batch_size=dims[-1], **grid_dims)
                    else:
                        t = tf.expand_dims(t, axis=-1)
                        # Grid based on number of outputs filters: (c_in, ?, ?, 1)
                        relevant_tensors[name] = tf.concat([viz_utils.image_grid(
                            t[:, :, :, k, :], batch_size=dims[-1], **grid_dims) for k in range(dims[2])], axis=0)
                        # Grid based on number of inputs filter
                        grid_dims = viz_utils.get_grid_size(dims[2])
                        relevant_tensors[name] = viz_utils.image_grid(relevant_tensors[name], batch_size=dims[2], **grid_dims)
                # Activations have shape (batch, w, h c_out)
                # We normalize them over (w, h)
                # We made them into a grid first based on c_out, then based on batch_size
                # Final tensor has shape (1, ?, ?, 1)
                elif display_type == 'activations':
                    # Normalize across channels and batch
                    t = tf.transpose(t, (3, 1, 2, 0))
                    t -= tf.reduce_min(t, axis=(1, 2), keep_dims=True)
                    t /= tf.reduce_max(t, axis=(1, 2), keep_dims=True) + 1e-8
                    # Grid based on number of outputs filters: (bath, ?, ?, 1)
                    relevant_tensors[name] = viz_utils.image_grid(t, batch_size=dims[-1], **grid_dims)
                    # Grid based on the batch_size
                    relevant_tensors[name] = tf.transpose(relevant_tensors[name], (3, 1, 2, 0))
                    n = self.config['batch_size'] if 'batch_size' in self.config else 1
                    grid_dims = viz_utils.get_grid_size(n)
                    relevant_tensors[name] = viz_utils.image_grid(relevant_tensors[name], batch_size=n, **grid_dims)
                # summaries have already been formatted to shape (1, ?, ?, 1 or 3)
                elif display_type == 'summaries':
                    t -= tf.reduce_min(t)
                    t /= tf.reduce_max(t) + 1e-8
                    relevant_tensors[name] = t
                else:
                    print('Warning: unknown display_type %s for tensor %s' % (display_type, name))
            # Plot
            with self.get_session('test', 
                                  restore_model_dir=restore_model_dir, 
                                  restore_model_scopes=restore_model_scopes,
                                  verbose=verbose) as sess:
                viz_utils.display_weights(sess.run(relevant_tensors), figwidth=figwidth) 
                
                
    def get_embeddings(self, 
                       embedding_layer_name, 
                       restore_model_dir, 
                       restore_model_scopes=None,
                       mode='test',
                       flatten=True,
                       verbose=False):
        """Load a checkpoint returns the relevant tensors on the whole dataset.
        
        Args:
            embedding_layer_name: The name of the layer containing the embeddings, or a list of layer names
            mode: Data split to use. One of val or train
            flatten: If True, output the flattened layers
            restore_model_dir: Directory to restore checkpoint from
            restore_model_scopes: Which scopes to restore, defaults to all
            figwidth: The width of the matplotlib figure
            verbose: verbose output
        """
        assert mode in ['test', 'val']
        with self.graph.as_default():
            if mode == 'val':
                self.validation_pass(reuse=False, verbose=verbose > 1)
            elif mode == 'test':
                self.test_pass(reuse=False, verbose=verbose > 1)
            if isinstance(embedding_layer_name, str):
                embedding_layer_name = [embedding_layer_name]
               
            # Get relevant tensors
            embeddings = [self.graph.get_tensor_by_name(x) for x in embedding_layer_name]
            if flatten:
                embeddings = [tf.layers.flatten(e) if len(e.get_shape()) > 1 else e for e in embeddings]
            if verbose > 1:
                for n, e in zip(embedding_layer_name, embeddings):
                    print('   Tensor \033[36m%s\033[0m, shape: %s' % (n, e.get_shape())) 
                
            # Precompute embedding matrix
            with self.get_session('test', 
                                  restore_model_dir=restore_model_dir, 
                                  restore_model_scopes=restore_model_scopes,
                                  verbose=verbose > 1) as sess:
                E = []
                try:
                    num_steps = 0
                    while 1:
                        # First step
                        if num_steps == 0:
                            E = sess.run(embeddings)
                        # Otherwise concatenate
                        else:
                            embeddings_ = sess.run(embeddings)
                            for i, x in enumerate(embeddings_):
                                E[i] = np.concatenate([E[i], x], axis=0)
                        if verbose: print('\r   \033[33m(embed)\033[0m step %08d' % num_steps, end='')
                        num_steps += 1
                except tf.errors.OutOfRangeError:
                    pass
                except KeyboardInterrupt:
                    print('Keyboard interrupted')
                # Return
                return E[0] if len(E) == 1 else E
            
    def get_weights(self, 
                    input_shape,
                    embedding_layer_name, 
                    restore_model_dir,
                    restore_model_scopes=None,
                    exclude_restore_model_scopes=None, 
                    replace_restore_model_scopes=None,
                    config={},
                    verbose=False):
        """Load a checkpoint returns the relevant tensors on the whole dataset.
        
        Args:
            input_shape: Shape of image inputs for this network
            embedding_layer_name: The name of the layer containing the embeddings, or a list of layer names
            restore_model_dir: Directory to restore checkpoint from
            restore_model_scopes: Which scopes to restore, defaults to all
            config: Feed-forward pass configuratoin
            verbose: verbose output
        """
        with self.graph.as_default():
            # basic graph
            inputs = {}
            inputs['image'] = tf.zeros(input_shape)
            self.feed_forward(inputs, config, is_chief=False)
            
            # Get weights
            if isinstance(embedding_layer_name, str):
                embedding_layer_name = [embedding_layer_name]
            embeddings = [self.graph.get_tensor_by_name(x) for x in embedding_layer_name]
            
            if verbose:
                for n, e in zip(embedding_layer_name, embeddings):
                    print('   Tensor \033[36m%s\033[0m, shape: %s' % (n, e.get_shape())) 
                
            # Load weights
            with self.get_session('test', 
                                  restore_model_dir=restore_model_dir, 
                                  restore_model_scopes=restore_model_scopes,
                                  exclude_restore_model_scopes=exclude_restore_model_scopes, 
                                  replace_restore_model_scopes=replace_restore_model_scopes,
                                  verbose=verbose) as sess:
                E = sess.run(embeddings)
                return E[0] if len(E) == 1 else E

            
def eval_accuracy(sess, accuracy_ops, num_samples, verbose=True):
    """Eval the given operation (return accuracy and number of samples) until 
       the input iterator hits a OutOfRange Error.
       
       Args:
           sess: A session
           accuracy_op: A list of tensors containing cumulatives accuracies
           num_samples: The total number of samples that the given accuracies were computed for
           verbose: verbosity level
        
       Returns:
           accuracy_: List of the same length as accuracy_ops containing the final normalized accuracy values
           num_samples_: The number of samples evaluated
       """
    try:
        num_steps_ = 0
        num_samples_ = 0
        accuracy_ = np.zeros((len(accuracy_ops),), dtype=np.float32)
        while 1:
            if verbose and num_samples_ % 50 == 0:
                print('\r   \033[33m(eval)\033[0m step %d' % num_steps_, end=' ' * 10)
            acc_, num_ = sess.run([accuracy_ops, num_samples])
            accuracy_ += acc_
            num_samples_ += num_
            num_steps_ += 1
    except tf.errors.OutOfRangeError:
        pass
    if verbose: print()
    accuracy_ /= float(num_samples_)
    return accuracy_, num_samples_