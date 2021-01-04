#!/usr/bin/env python3

import sys
sys.path.append("..")
import os
import argparse
from termcolor import colored

from path_denoise_3d.detector_configuration import *
from path_denoise_3d.preprocessing import *
from pandas_from_json import hits_array_from_parquet, coordinates_dataframe_from_parquet
from models.CnnDenoisingModel import CnnDenoisingModel
import matplotlib.pyplot as plt
import numpy as np
import histogram as hm

def main():
    args = parse_arguments()
    print(args)

    subroutine = get_subroutine(args)
    subroutine(args)


def parse_arguments():
    """
    Parse CLI arguments and return an object containing the values.
    """

    # Create main program parser
    parser = argparse.ArgumentParser(description="CRTC-JLab Machine Learning 3D Track Denoising CLI")
    subparsers = parser.add_subparsers()
    subparsers.required = True
    subparsers.dest = "subprogram"

    # Create training subprogram parser
    parser_train = subparsers.add_parser("train", help="Train a model, perform testing, and serialize it.")
    parser_train._action_groups.pop()
    parser_train_required_args = parser_train.add_argument_group("Required arguments")
    parser_train_optional_args = parser_train.add_argument_group("Optional arguments")
    parser_train_required_args.add_argument("--training-data", "-d", required=True,
                                            help="Path to the PARQUET file containing the training data.",
                                            dest="data_file_path")

    # parser_train_required_args.add_argument("--out-model", "-m", required=True,
    #                                         help="Name of the file in which to save the model.",
    #                                         dest="output_model_path")
    
    parser_train_required_args.add_argument("--results-dir", "-r", required=True,
                                            help="Path to the directory to store the model produced and related results.",
                                            dest="results_dir")

    parser_train_optional_args.add_argument("--quantization", "-q", choices=["21", "32", "43"], default=None,
                                            required=False, help="Path to the directory containing the testing data.",
                                            dest="quantization")
    parser_train_optional_args.add_argument("--noise-percentage", "-n", default=0.45, required=False,
                                            help="Percentage of noise to add to dataset",
                                            dest="noise_percentage")
    parser_train_optional_args.add_argument("--padding", "-p", default=6, required=False,
                                            help="Padding to add to the dataset",
                                            dest="padding")
    parser_train_optional_args.add_argument("--epochs", "-e", required=False, type=int, default="10",
                                            help="How many training epochs to go through.", dest="epochs")
    parser_train_optional_args.add_argument("--batch-size", "-b", required=False, type=int, default="8",
                                            help="Size of the training batch.", dest="batch_size")

    # Create evaluation subprogram parser
    parser_test = subparsers.add_parser("test", help="Load a model for testing.")
    parser_test._action_groups.pop()
    parser_test_required_args = parser_test.add_argument_group("Required arguments")
    parser_test_optional_args = parser_test.add_argument_group("Optional arguments")
    parser_test_required_args.add_argument("--testing-data", "-d", required=True,
                                           help="Path to the PARQUET file containing the testing data.",
                                           dest="data_file_path")
    parser_test_required_args.add_argument("--model", "-m", required=True,
                                           help="The name of the file from which to load the model.", dest="model_path")
   
    parser_test_required_args.add_argument("--results-dir", "-r", required=True,
                                            help="Path to the directory to store results.",
                                            dest="results_dir")
    parser_test_optional_args.add_argument("--quantization", "-q", choices=["21", "32", "43"], default=None,
                                           required=False, help="Path to the directory containing the testing data.",
                                           dest="quantization")
    parser_test_optional_args.add_argument("--noise-percentage", "-n", default=0.45, required=False,
                                            help="Percentage of noise to add to dataset",
                                            dest="noise_percentage")
    parser_test_optional_args.add_argument("--padding", "-p", default=6, required=False,
                                            help="Padding to add to the dataset",
                                            dest="padding")
    parser_test_optional_args.add_argument("--batch-size", "-b", required=False, type=int, default="8",
                                           help="Size of the evaluation batch.", dest="batch_size")

    # Create prediction subprogram parser
    parser_predict = subparsers.add_parser("predict", help="Load a model and use it for predictions.")
    parser_predict._action_groups.pop()
    parser_predict_required_args = parser_predict.add_argument_group("Required arguments")
    parser_predict_optional_args = parser_predict.add_argument_group("Optional arguments")

    parser_predict_required_args.add_argument("--prediction-data", "-d", required=True,
                                              help="Path to the PARQUET file containing the prediction data.",
                                              dest="data_file_path")

    parser_predict_required_args.add_argument("--model", "-m", required=True,
                                              help="The name of the file from which to load the model.",
                                              dest="model_path")

    parser_predict_required_args.add_argument("--results-dir", "-r", required=True,
                                            help="Path to the directory to store results.",
                                            dest="results_dir")

    parser_predict_optional_args.add_argument("--quantization", "-q", choices=["21", "32", "43"], default=None,
                                              required=False, help="Path to the directory containing the testing data.",
                                              dest="quantization")

    parser_predict_optional_args.add_argument("--batch-size", "-b", required=False, type=int, default="32",
                                              help="Size of the prediction batch.", dest="prediction_batch_size")

    parser_predict_optional_args.add_argument("--padding", "-p", default=6, required=False,
                                            help="Padding to add to the dataset",
                                            dest="padding")

    return parser.parse_args()


def read_input_data(input_type, args) -> dict:
    """
    Reads a dataset in Parquet format and preprocesses the data by generating noisy data and doing a training/test split.

    Args:
        input_type: Type of reading and processing of the input for training, evaluation, or prediction.
        args: The object that contains all the parsed CLI arguments.

    Returns:
        A dictionary containing the read and processed input data.
    """

    total_planes = 10 if args.quantization is None else int(args.quantization)

    # Assign Z offset ranges based on the quantization parameter
    z_range_offset = None
    if args.quantization is None:
        z_range_offset = None
    elif args.quantization == "21":
        z_range_offset = Z_OFFSET_RANGES_21
    elif args.quantization == "32":
        z_range_offset = Z_OFFSET_RANGES_32
    elif args.quantization == "43":
        z_range_offset = Z_OFFSET_RANGES_43

    if input_type == "train":
        # Read training and testing data
        data = hits_array_from_parquet(args.data_file_path, z_range_offset)[0]
        x_train, x_test, y_train, y_test = preprocess_data_training(data, total_planes, args.padding, args.noise_percentage)

        return {
            "training": {"data": x_train, "labels": y_train, "epochs": args.epochs, "batch_size": args.batch_size},
            "testing": {"data": x_test, "labels": y_test, "batch_size": args.batch_size},
            "configuration": {"total_planes": total_planes, "rings_per_plane": RINGS_PER_PLANE,
                              "pads_per_ring": PADS_PER_RING + args.padding}
        }

    elif input_type == "test":
        # Read testing data
        data = hits_array_from_parquet(args.data_file_path, z_range_offset)[0]
        x_test, y_test = preprocess_data_testing(data, total_planes, args.padding, args.noise_percentage)

        return {
            "testing": {"data": x_test, "labels": y_test, "batch_size": args.batch_size},
            "configuration": {"total_planes": total_planes, "rings_per_plane": RINGS_PER_PLANE,
                              "pads_per_ring": PADS_PER_RING + args.padding}
        }

    elif input_type == "predict":
       data = hits_array_from_parquet(args.data_file_path, z_range_offset)[0]
       data = preprocess_data_predicting(data, total_planes, args.padding)

       return {
           "prediction": {"data": data, "quantization": z_range_offset, "input": args.data_file_path},
           "configuration": {"total_planes": total_planes, "rings_per_plane": RINGS_PER_PLANE,
                                "pads_per_ring": PADS_PER_RING + args.padding}
        }

    else:
        print(colored("Error: Wrong input type.", "red"))
        quit()

def plot_random_predicted_events(results_dir, ground_truth, noisy, prediction, num_random_events, seed = 22):
    """
    Plots random events after clearing them

    Args:
        results_dir (string): Directory to store the generated images
        ground_truth (numpy array): Numpy array with the actual tracks
        noisy (numpy array): Numpy array with the noisy tracks given as input
        prediction (numpy array): Numpy array with the tracks predicted
        num_random_events (int): Number of random events to print
        seed (int). [Optional]: Seed for the random number generator
    """
    img_shape = ground_truth.shape[-3], ground_truth[0].shape[-2]
    img_shape = prediction.shape[-3], prediction[0].shape[-2]
    img_shape = noisy.shape[-3], noisy[0].shape[-2]
    img_indexes = np.random.choice(prediction.shape[0], num_random_events, replace = False)

    for i, index in enumerate(img_indexes):
        plt.imsave(results_dir+'correct'+str(i)+'.png', ground_truth[index].reshape(img_shape))
        plt.imsave(results_dir+'noisy'+str(i)+'.png', noisy[index].reshape(img_shape))
        plt.imsave(results_dir+'denoised'+str(i)+'.png', (prediction[index].reshape(img_shape)>0.5).astype(int))




def plot_accuracy_histogram(testing_metrics):
    """
    Plots the accuracy histograms, hits and noise

    Args:
        training_metrics (dictionary): Dictionary outputted by the training function
    """

    ground_truth = testing_metrics["truth"]
    ground_truth = ground_truth.reshape((-1, ground_truth.shape[-3] * ground_truth.shape[-2]))
    predictions = testing_metrics["predictions"]
    predictions = predictions.reshape((-1, predictions.shape[-3] * predictions.shape[-2]))

    results_dir = testing_metrics["results_dir"]

    hits_stats = hm.plot_hits(results_dir+'hits_histogram.png', predictions, ground_truth)
    noise_stats = hm.plot_noise(results_dir+'noise_histogram.png', predictions, ground_truth)
    cases = hits_stats["num"]
    hits_max = hits_stats["max"]
    hits_min = hits_stats["min"]

    print(f'{colored("Total number of cases:", "blue")} {cases}')
    print(f'{colored("Hits Minimum value(%):", "blue")} {hits_min}')
    print(f'{colored("Hits Maximum value(%)::", "blue")} {hits_max}')

    noise_max = noise_stats["max"]
    noise_min = noise_stats["min"]
    print(f'{colored("Noise Minimum value(%):", "blue")} {noise_min}')
    print(f'{colored("Noise Maximum value(%)::", "blue")} {noise_max}')

    with open(results_dir + 'testing_report','a+') as f:
        f.write('Total number of cases: '+ str(cases) + '\n')
        f.write('Hits Minimum value(%): ' + str(hits_min) + '\n')
        f.write('Hits Maximum value(%): ' + str(hits_max) + '\n')
        f.write('Noise Minimum value(%): ' + str(noise_min) + '\n')
        f.write('Noise Maximum value(%): ' + str(noise_max) + '\n')



def plot_train_val_graph(training_metrics):
    """
    Plots the training/validation loss versus epochs graph

    Args:
        training_metrics (dictionary): Dictionary outputted by the training function
    """
    training_loss_history = training_metrics["training_loss_history"]
    validation_loss_history = training_metrics["validation_loss_history"]
    results_dir = training_metrics["results_dir"]
    
    epoch_count = range(1, len(training_loss_history) + 1)
    plt.figure(1)

    plt.plot(epoch_count, training_loss_history, 'r--')
    plt.plot(epoch_count, validation_loss_history, 'b--')
    plt.yticks(np.arange(0, 1.20*max(training_loss_history[0], validation_loss_history[0]), step=max(training_loss_history[0], validation_loss_history[0])/20))
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(results_dir+'train_val_loss_graph.png')
    

def print_training_report(training_metrics):
    """
    Prints a report for the training of a machine learning model.

    Args:
        training_metrics: Dictionary that was outputted by the training function
    """

    training_loss = training_metrics["training_loss"]
    training_time = training_metrics["training_time"]
    training_results_dir = training_metrics["results_dir"]
    
    with open(training_results_dir + 'training_report','w+') as f:
        f.write("Training loss: " + str(training_loss) +'\n')
        f.write("Training time: " + str(training_time) +'\n')
    print("\nTraining Report")
    print("================================")
    print(f'{colored("Training loss:", "blue")} {training_loss}')
    print(f'{colored("Training time:", "blue")} {training_time}s')



def print_testing_report(testing_metrics):
    """
    Prints a report for the training of a machine learning model.

    Args:
        training_dict: Dictionary that was outputted by the training function
    """

    testing_loss = testing_metrics["testing_loss"]
    testing_prediction_time = testing_metrics["testing_prediction_time"]
    testing_results_dir = testing_metrics["results_dir"]

    with open(testing_results_dir + 'testing_report','w+') as f:
        f.write("Testing loss: " + str(testing_loss) +'\n')
        f.write("Testing prediction time: " + str(testing_prediction_time) +'\n')

    print("\nTesting Report")
    print("================================")
    print(f'{colored("Testing loss:", "blue")} {testing_loss}')
    print(f'{colored("Testing prediction time:", "blue")} {testing_prediction_time}s')


def train_model(args):
    """
    Trains the model with the input data specified in the CLI arguments.

    Args:
        args: The object that contains all the parsed CLI arguments.
    """

    print(colored("\nReading input data...", "green"))
    input_dict = read_input_data("train", args)
    results_dir = args.results_dir.rstrip('/')+'/'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    model = CnnDenoisingModel(input_dict=input_dict)

    model.build_new_model()
    training_metrics = model.train(input_dict)
    training_metrics["results_dir"] = results_dir

    model.save_model(results_dir+'cnn_autoenc')
    print_training_report(training_metrics)
    plot_train_val_graph(training_metrics)
    print(colored(f'\nSaving training results to {results_dir}\n', "green"))


def test_model(args):
    """
    Tests a model with the input data specified in the CLI arguments.

    Args:
        args: The object that contains all the parsed CLI arguments.
    """

    print(colored("\nReading input data...", "green"))
    input_dict = read_input_data("test", args)
    results_dir = args.results_dir.rstrip('/')+'/'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    model = CnnDenoisingModel(input_dict=input_dict)

    model.load_model(args.model_path)

    testing_metrics = model.test(input_dict)
    testing_metrics["results_dir"] = results_dir

    print_testing_report(testing_metrics)
    plot_accuracy_histogram(testing_metrics)
    plot_random_predicted_events(results_dir, input_dict["testing"]["labels"], input_dict["testing"]["data"], testing_metrics["predictions"], 6)
    print(colored(f'\nSaving testing results to {results_dir}\n', "green"))

def save_prediction_results(predict_metrics):
    """
    Creates a new dataset clean from the noisy hits and stores it into parquet format

    Args:
        predict_metrics (dictionary): Dictionary with the required arguments to create the cleaned
        dataset
    """

    data = predict_metrics["input_data"]
    predictions = predict_metrics["predictions"]
    path_to_hits_df = predict_metrics["path_to_df"]
    num_planes = predict_metrics["num_planes"]
    quantization = predict_metrics["quantization"]
    results_dir = predict_metrics["results_dir"]


    predictions = predictions.reshape((-1, num_planes, (predictions.shape[-3])//num_planes, predictions.shape[-2]))
    data = data.reshape((-1, num_planes, (data.shape[-3])//num_planes, data.shape[-2]))

    bad_tuples = np.where(predictions!= data)

    df = coordinates_dataframe_from_parquet(path_to_hits_df, quantization)
    if len(bad_tuples) == 3:
        df = df[~df[['plane', 'ring', 'pad']].apply(tuple, 1).isin(bad_tuples)]
    elif len(bad_tuples) == 4:
        df = df[~df[['eventid', 'plane', 'ring', 'pad']].apply(tuple, 1).isin(bad_tuples)]
    df.to_parquet(results_dir+'denoised_data.parquet', index=False)
    print(colored(f'\nSaving prediction results to {results_dir}\n', "green"))

def predict(args):
    """
    Uses a model for predicting values for input data read from a Parquet file.

    Args:
        args: The object that contains all the parsed CLI arguments.
    """
    print(colored("\nReading input data...", "green"))
    input_dict = read_input_data("predict", args)
    results_dir = args.results_dir.rstrip('/')+'/'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    model = CnnDenoisingModel(input_dict=input_dict)

    model.load_model(args.model_path)
    predict_metrics = model.predict(input_dict)
    predict_metrics["results_dir"] = results_dir
    
    predict_metrics["input_data"] = input_dict["prediction"]["data"]
    predict_metrics["num_planes"] = input_dict["configuration"]["total_planes"]
    predict_metrics["path_to_df"] = input_dict["prediction"]["input"]
    predict_metrics["quantization"] = input_dict["prediction"]["quantization"]
    save_prediction_results(predict_metrics)


def get_subroutine(args):
    """
    Processes the CLI subprogram utilized and return the matching function for training, testing, or prediction.

    Args:
        args: The object that contains all the parsed CLI arguments.

    Returns:
        The functions corresponding the the CLI subprogram utilized.
    """

    if args.subprogram == "train":
        print(colored("Executing training subprogram.", "green"))
        return train_model
    elif args.subprogram == "test":
        print(colored("Executing testing subprogram.", "green"))
        return test_model
    elif args.subprogram == "predict":
        print(colored("Executing prediction subprogram.", "green"))
        return predict
    else:
        print(colored("Fatal Error: Wrong subprogram specified.", "red"))
        quit()


if __name__ == "__main__":
    main()
