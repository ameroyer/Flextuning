
*ReadMe and instructions: Work in progress*

## Introduction
Implementation of Flextuning described in [A Flexible Selection Scheme for Minimum-Effort Transfer Learning, WACV 2020](https://openaccess.thecvf.com/content_WACV_2020/papers/Royer_A_Flexible_Selection_Scheme_for_Minimum-Effort_Transfer_Learning_WACV_2020_paper.pdf)

```
@article{flextuning,
		        author = {Royer, Am\'{e}lie and Lampert, Christoph H.},
		        title = {A Flexible Selection Scheme for Minimum-Effort Transfer Learning},
		        journal = {Winter Conference on Applications of Computer Vision (WACV)},
		        year = {2020}
		      }
```


## Quick usage
```bash
# Requires Tensorflow (tested with 1.12 / 1.14)

# Download Mnist source/target domains from 
wget ist.ac.at/~aroyer/Models/Data.zip
# Alternatively you can compute them (see input_pipeline.ipynb)


# Run only the first time
mkdir results
mkdir txt_logs
unzip Data.zip

# Main
python3 main.py mnist --all
```

Will run one repeat of all tuning experiments on MNIST -> target domains.
Results of experiments are printed in terminal + saved in a dictionary (in results/). Additional logs are output in txt_logs/ (just text files) and log/ (Tensorboard summaries). See `main.py`.


## Codebase
#### Datasets:
  * `input_pipeline.ipynb` : Visualize the datasets + Convert them to TFRecords format for convenience. Also pretrain a "base model" on the source domain that will be used as initialization for tuning.
  * `setting.py` : Defines the different source and target domains for convenience + if it is run at main, it will create susampled ratios of each datasets.
  * `include/dataset_utils.py`: Functions to parse the datasets + image transformation for the target domains. Use low-level functions defined in `include/tfrecords_utils.py` and `include/preprocess_utils.py`.


#### Classifiers:
  * `net_utils.py` : Architectures (Note: The second half of the file is only used for experiments described in section 5.4 in the paper. You can also ignore the file `include/graph_utils.py`, it is also only really useful for section 5.4 for optional preprcessing when feeding outputs from the preprocessing module to the classifier)
  * `include/graph_manager.py` : It defines two things: First a "Setting" object, which essentially a wrapper to define source/target datasets. and a "Classifier" object which defines the training loop. it comes with "train" and "test" functions: This part is probably the most "Tensorflow specific" part of the code. For convenience, subclasses of the Classifier base class are then defined in include/classifiers.py



#### Run experiments:
  * `main.py`: Basically gathers the whole experiment setups
  * `include/run_exp.py`: Given a classfier, and a list of source/domain datasets, it takes care of training/evaluating all different finetuned/flextuned models and save logs.


#### Fast and Faster-flex variants
Which unit to "flextune" ? The two faster selectio criteria are explored in `selection_criterion.ipynb`
