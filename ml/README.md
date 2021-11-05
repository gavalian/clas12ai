# Using Artificial Intelligence for Particle Track Identification in the CLAS12 Detecto

In this project machine learning is used to classify particle tracks from data generated from drift chambers in Jefferson Laboratory CLAS12 detector. The provided script can be used to train different models on pre-classified data and then use the trained models on new data on-line to help with the reconstruction process.


# Requirements and Installation
This software is developed in python 3 and requires the following libraries:
* Python 3.7+ 
* Tensorflow
* Numpy
* Matplotlib
* scikit-learn

To simplify the installation process a ```YAML``` file is provided that can be used with ```anaconda``` dependency manager system to automatically install all dependencies.

To install dependencies using [anaconda](https://www.anaconda.com/), first install anaconda and then use the following command:
```bash
conda env create -f ml.yaml
```
This should install all required dependencies for our software.

# Usage
To use the provided script first enable the conda environment you just created (if used anaconda for that):
```bash
conda activate ml
```

The software provides three subprograms:
* ```train``` , for training a new model on provided data
* ```test``` , to test an existing model on a new dataset and get accuracy metrics
* ```predict```, to use an existing model on a dataset and store the results generated

All input data are expected to be in [lsvm](https://www.cs.cornell.edu/people/tj/svm_light/) format with 1 as the label for a valid track and 0 for an invalid one. Then the id of each wire that indicated a detection, followed by a colon and an 1 (```id: 1```). Wires with no detection can also be included with id colon 0 (```id: 0```). <br>
An example is presented [here](https://userweb.jlab.org/~gavalian/ML/2021/Denoise/dc_denoise_one_track_1.lsvm).

## Train
To train a new model run the train command. Its arguments are provided below:
```bash
$python3 ml-cli.py train -h
usage: ml-cli.py train [-h] --training-file TRAINING_FILE_PATH --testing-file
                       TESTING_FILE_PATH --num-features {6,36,4032}
                       --out-model OUTPUT_MODEL_PATH --model-type
                       {cnn,mlp,et,rnn} [--epochs TRAINING_EPOCHS]
                       [--batchSize TRAINING_BATCH_SIZE]
                       [--testing-batchSize EVALUATION_BATCH_SIZE]

Required arguments:
  --training-file TRAINING_FILE_PATH, -t TRAINING_FILE_PATH
                        Path to the file containing the training data.
  --testing-file TESTING_FILE_PATH, -e TESTING_FILE_PATH
                        Path to the file containing the testing data.
  --num-features {6,36,4032}, -f {6,36,4032}
                        Path to the directory containing the testing data.
  --out-model OUTPUT_MODEL_PATH, -m OUTPUT_MODEL_PATH
                        Name of the file in which to save the model.
  --model-type {cnn,mlp,et,rnn}
                        The type of the model to train.

Optional arguments:
  --epochs TRAINING_EPOCHS
                        How many training epochs to go through.
  --batchSize TRAINING_BATCH_SIZE
                        Size of the training batch.
  --testing-batchSize EVALUATION_BATCH_SIZE
                        Size of the evaluation batch.
```
For example, to train an mlp model for 10 epochs on a new dataset ```train-set.lsvm```, store it as ```mlp.p``` and validate on ```validation-set.lsvm``` run:
```bash
python3 ml-cli.py train -t train-set.lsvm --epochs 10 -e validation-set.lsvm -m mlp.p --model-type mlp -f 6
```

## Test
The testing subcommand expects a model trained by the train subcommand, a dataset to test it on and a directory path to store the evaluation results. You can get a complete list of its parameters by passing the -h parameter.

```bash
$python3 ml-cli.py test -h
usage: ml-cli.py test [-h] --testing-file TESTING_FILE_PATH --num-features
                      {6,36,4032} --model MODEL_PATH --model-type
                      {cnn,mlp,et,rnn} [--batchSize EVALUATION_BATCH_SIZE]

Required arguments:
  --testing-file TESTING_FILE_PATH, -e TESTING_FILE_PATH
                        Path to the file containing the testing data.
  --num-features {6,36,4032}, -f {6,36,4032}
                        Path to the directory containing the testing data.
  --model MODEL_PATH, -m MODEL_PATH
                        The name of the file from which to load the model.
  --model-type {cnn,mlp,et,rnn}
                        The type of the model to load.

Optional arguments:
  --batchSize EVALUATION_BATCH_SIZE
                        Size of the evaluation batch.
```
For example, to load a mlp model ```mlp.p``` and run on a new dataset ```test-set.lsvm``` run:
```bash
python3 ml-cli.py test -e test-set.lsvm -f 6 -m mlp.p --model-type mlp
```
## Datasets

The datasets used for model studies can be found [here](https://userweb.jlab.org/~gavalian/ML/2021/Classifier/) <br>


# JLab Machine Learning Project
This repository contains code pertaining to the ML project for JLab.

# Requirements
* Anaconda (tested with Python 3.7)

# Installation
Create a new Anaconda environment using the YML file provided:

`conda env create -f conda_environment.yml`

# Usage
**Make sure to activate the conda environment.**

Execute `ml-cli.py` under `src/` with the necessary CLI arguments.

To see the available CLI parameters execute `./ml-cli.py -h`

The script makes use of subcommands. 

To get help for the `train` subcommand for example, execute `./ml-cli.py train -h`.
