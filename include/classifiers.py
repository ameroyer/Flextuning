from functools import partial

from . import graph_utils
from . import net_utils

from .graph_manager import Classifier

class _MnistNetClassifier(Classifier):
    """Defines a mnist_net based Classifier object."""
    model_scopes = ['mnist_net/layer_%d' % i for i in [1, 2, 3, 4]]
    
    def __init__(self, 
                 get_dataset_fn,
                 preprocess_fn=None,
                 trainable_scopes=None,
                 config={}):
        """Args:
            get_dataset_fn: Dataset function
            preprocess_fn: Preprocessing function
            trainable_scopes: Scopes to train. Defaults to None ie all trainable variables
            config: Additional keyword configuration.
        """
        Classifier.__init__(self, 
                            get_dataset_fn, 
                            net_utils.mnist_net, 
                            preprocess_fn=preprocess_fn,
                            transition_fn=graph_utils.mnist_transition,
                            trainable_scopes=trainable_scopes, 
                            scope_name='mnist_net',                           
                            config=config)   
    
class MNISTClassifier(_MnistNetClassifier):  
    def __init__(self, get_dataset_fn, trainable_scopes=None,  config={}):
        super(MNISTClassifier, self).__init__(get_dataset_fn, 
                                              preprocess_fn=graph_utils.mnist_preprocess,
                                              trainable_scopes=trainable_scopes, 
                                              config=config)
        
    
class SpatialMNISTClassifier(_MnistNetClassifier):
    def __init__(self, get_dataset_fn, trainable_scopes=None, config={}):
        super(SpatialMNISTClassifier, self).__init__(get_dataset_fn, 
                                                     preprocess_fn=graph_utils.mnist_spatial_preprocess,
                                                     trainable_scopes=trainable_scopes,
                                                     config=config)    

        
class _InceptionClassifier(Classifier) :
    """Defines a inception based Classifier object."""
    model_scopes = ['imagenet_net/InceptionV2/block1', 'imagenet_net/InceptionV2/block2', 
                    'imagenet_net/InceptionV2/block3a', 'imagenet_net/InceptionV2/block3b', 
                    'imagenet_net/InceptionV2/block3c', 'imagenet_net/InceptionV2/block4a', 
                    'imagenet_net/InceptionV2/block4b', 'imagenet_net/InceptionV2/block4c',
                    'imagenet_net/InceptionV2/block4d', 'imagenet_net/InceptionV2/block4e', 
                    'imagenet_net/InceptionV2/block5a', 'imagenet_net/InceptionV2/block5b', 
                    'imagenet_net/InceptionV2/logits']
    
    def __init__(self, 
                 get_dataset_fn, 
                 feed_forward_fn=net_utils.inception_net,
                 preprocess_fn=None,
                 trainable_scopes=None,
                 config={}):
        Classifier.__init__(self, 
                            get_dataset_fn, 
                            feed_forward_fn,
                            preprocess_fn=preprocess_fn,
                            transition_fn=graph_utils.imagenet_transition,
                            trainable_scopes=trainable_scopes, 
                            scope_name='imagenet_net',                           
                            config=config)           
        
class ILSVRCClassifier(_InceptionClassifier):  
    def __init__(self, get_dataset_fn, trainable_scopes=None, config={}):
        super(ILSVRCClassifier, self).__init__(get_dataset_fn,
                                               net_utils.ilsvrc_net, 
                                               preprocess_fn=graph_utils.imagenet_preprocess,
                                               trainable_scopes=trainable_scopes, config=config)
        
class SpatialILSVRCClassifier(_InceptionClassifier):
    def __init__(self, 
                 get_dataset_fn, 
                 trainable_scopes=None,
                 config={}):
        super(SpatialILSVRCClassifier, self).__init__(get_dataset_fn, 
                                                      net_utils.ilsvrc_net, 
                                                      preprocess_fn=graph_utils.ilsvrc_spatial_preprocess,
                                                      trainable_scopes=trainable_scopes, 
                                                      config=config)
        
class ColorILSVRCClassifier(_InceptionClassifier):  
    def __init__(self, 
                 get_dataset_fn,
                 trainable_scopes=None,
                 config={}):
        super(ColorILSVRCClassifier, self).__init__(get_dataset_fn, 
                                                    net_utils.ilsvrc_net, 
                                                    preprocess_fn=graph_utils.color_preprocess,
                                                    trainable_scopes=trainable_scopes,
                                                    config=config)
        
    
class PACSClassifier(_InceptionClassifier):
    def __init__(self, get_dataset_fn, trainable_scopes=None, config={}):  
        super(PACSClassifier, self).__init__(get_dataset_fn,
                                             net_utils.pacs_net, 
                                             preprocess_fn=graph_utils.imagenet_preprocess,
                                             trainable_scopes=trainable_scopes, config=config)
        
    
class PACSFromImagenetClassifier(_InceptionClassifier):
    def __init__(self, get_dataset_fn, trainable_scopes=None, config={}):  
        super(PACSFromImagenetClassifier, self).__init__(get_dataset_fn,
                                                         net_utils.pacs_from_imagenet_net, 
                                                         preprocess_fn=graph_utils.imagenet_preprocess,
                                                         trainable_scopes=trainable_scopes, config=config)
        
        
class _CIFARNetClassifier(Classifier):
    """Defines a cifar_net based Classifier object."""
    model_scopes = ['cifar_net/layer_%d' % i for i in range(1, 8)]
    
    def __init__(self, 
                 get_dataset_fn,
                 num_channels=3,
                 preprocess_fn=None,
                 trainable_scopes=None,
                 config={}):
        """Args:
            get_dataset_fn: Dataset function
            preprocess_fn: Preprocessing function
            trainable_scopes: Scopes to train. Defaults to None ie all trainable variables
            config: Additional keyword configuration.
        """
        Classifier.__init__(self, 
                            get_dataset_fn, 
                            net_utils.cifar_net, 
                            preprocess_fn=preprocess_fn,
                            transition_fn=partial(graph_utils.cifar_transition, num_channels=num_channels),
                            trainable_scopes=trainable_scopes, 
                            scope_name='cifar_net',                           
                            config=config)   
    
class CIFARClassifier(_CIFARNetClassifier):  
    def __init__(self, 
                 get_dataset_fn, 
                 trainable_scopes=None,
                 config={}):
        super(CIFARClassifier, self).__init__(get_dataset_fn,
                                              num_channels=3,
                                              preprocess_fn=graph_utils.imagenet_preprocess,
                                              trainable_scopes=trainable_scopes, config=config)
    
class QuickDrawClassifier(_CIFARNetClassifier):  
    def __init__(self, 
                 get_dataset_fn, 
                 trainable_scopes=None,
                 config={}):
        super(QuickDrawClassifier, self).__init__(get_dataset_fn, 
                                                  num_channels=1,
                                                  preprocess_fn=graph_utils.mnist_preprocess,
                                                  trainable_scopes=trainable_scopes, config=config)