# CLAS12 Track Denoising Using AutoEncoders

In this project machine learning is used to denoise the data generated from drift chambers in Jefferson Laboratory CLAS12 detector. The provided script can be used to train and test AutoEncoder models on "noisy" and "denoised" detector data. The resulting output can be used to more efficiently detect valid tracks in drift chambers.


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
usage: ml-cli.py train [-h] --training-file TRAINING_FILE_PATH
                       --validation-file TESTING_FILE_PATH --results-dir
                       RESULTS_DIR [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                       [--network NN]

Required arguments:
  --training-file TRAINING_FILE_PATH, -t TRAINING_FILE_PATH
                        Path to the SVM file containing the training data.
  --validation-file TESTING_FILE_PATH, -v TESTING_FILE_PATH
                        Path to the SVM file containing the training data.
  --results-dir RESULTS_DIR, -r RESULTS_DIR
                        Path to the directory to store the model produced and
                        related results.

Optional arguments:
  --epochs EPOCHS, -e EPOCHS
                        How many training epochs to go through.
  --batch-size BATCH_SIZE, -b BATCH_SIZE
                        Size of the training batch.
  --network NN, -n NN   Neural network to use.
```
For example, to train a model for 10 epochs on a new dataset ```train-set.lsvm``` and store the generated model along with information related to the process to a new directory called ```train-results``` run:
```bash
python3 ml-cli.py train -t train-set.lsvm -e 10 -r train-results
```
The ```--network``` argument accepts a value in [0, 0a - 0g, 1, 2] which represent architectures defined in the paper.

## Test
The testing subcommand expects a model trained by the train subcommand, a dataset to test it on and a directory path to store the evaluation results. You can get a complete list of its parameters by passing the -h parameter. The ```--threshold``` sets the minimum value which is interpreted as a detection from a wire in the output of the denoising model.

```bash
$python3 ml-cli.py test -h

usage: ml-cli.py test [-h] --validation-file TESTING_FILE_PATH --model
                      MODEL_PATH --results-dir RESULTS_DIR
                      [--threshold THRESHOLD] [--batch-size BATCH_SIZE]

Required arguments:
  --validation-file TESTING_FILE_PATH, -v TESTING_FILE_PATH
                        Path to the SVM file containing the training data.
  --model MODEL_PATH, -m MODEL_PATH
                        The name of the file from which to load the model.
  --results-dir RESULTS_DIR, -r RESULTS_DIR
                        Path to the directory to store results.

Optional arguments:
  --threshold THRESHOLD
                        Threshold for valid track
  --batch-size BATCH_SIZE, -b BATCH_SIZE
                        Size of the evaluation batch.
```

## Predict

The predict subprogram accepts the same parameters as the test subprogram
but in addition to what test does it also outputs a dataset with the denoised output. A list of its parameters is shown below:

```bash
$python3 ml-cli.py predict -h

usage: ml-cli.py predict [-h] --prediction-file PREDICTION_FILE_PATH --model
                         MODEL_PATH --results-dir RESULTS_DIR
                         [--batch-size PREDICTION_BATCH_SIZE]
                         [--threshold THRESHOLD]

Required arguments:
  --prediction-file PREDICTION_FILE_PATH, -p PREDICTION_FILE_PATH
                        Path to the SVM file containing the prediction data.
  --model MODEL_PATH, -m MODEL_PATH
                        The name of the file from which to load the model.
  --results-dir RESULTS_DIR, -r RESULTS_DIR
                        Path to the directory to store results.

Optional arguments:
  --batch-size PREDICTION_BATCH_SIZE, -b PREDICTION_BATCH_SIZE
                        Size of the prediction batch.
  --threshold THRESHOLD
                        Threshold for valid track
```

## Datasets

The datasets used for model studies can be found [here](https://userweb.jlab.org/~gavalian/ML/2021/Denoise/) <br>
Dataset for the luminosity and threshold studies are [here](https://userweb.jlab.org/~gavalian/ML/2021/Denoise/luminocity_fixed/) 

## Running experiments presented in paper

To run most of the experiments presented in the paper and generate the respective plots with provide a script that automates this process in the '''experiments''' directory. 
```bash
cd experiments
bash ./run_experiments.sh
```
Should download all datasets and run the experiments. <br>
Note that the plots in section 8 require running the CLAS12 reconstruction software and are thus not provided.
Also, randomness is an integral of the Machine Learning (e.g. weight initialization) process, so the generated results might vary to a degree in comparison to the results presented in the paper. 
