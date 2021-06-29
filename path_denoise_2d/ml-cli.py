#!/usr/bin/env python3

import sys
sys.path.append("..")
import os
import argparse
from termcolor import colored

from process_input import *
# from pandas_from_json import hits_array_from_parquet, coordinates_dataframe_from_parquet
import matplotlib
matplotlib.use('pdf')
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
    parser = argparse.ArgumentParser(description="CRTC-JLab Machine Learning 2D Track Denoising CLI")
    subparsers = parser.add_subparsers()
    subparsers.required = True
    subparsers.dest = "subprogram"

    # Create training subprogram parser
    parser_train = subparsers.add_parser("train", help="Train a model, perform testing, and serialize it.")
    parser_train._action_groups.pop()
    parser_train_required_args = parser_train.add_argument_group("Required arguments")
    parser_train_optional_args = parser_train.add_argument_group("Optional arguments")

    parser_train_required_args.add_argument("--training-file", "-t", required=True,
                                            help="Path to the SVM file containing the training data.",
                                            dest="training_file_path")

    parser_train_required_args.add_argument("--validation-file", "-v", required=True,
                                            help="Path to the SVM file containing the training data.",
                                            dest="testing_file_path")

    parser_train_required_args.add_argument("--results-dir", "-r", required=True,
                                            help="Path to the directory to store the model produced and related results.",
                                            dest="results_dir")

    parser_train_optional_args.add_argument("--epochs", "-e", required=False, type=int, default="10",
                                            help="How many training epochs to go through.", dest="epochs")

    parser_train_optional_args.add_argument("--batch-size", "-b", required=False, type=int, default="8",
                                            help="Size of the training batch.", dest="batch_size")

    parser_train_optional_args.add_argument("--network", "-n", required=False, default="",
                                            help="Neural network to use.", dest="nn")

    # Create evaluation subprogram parser
    parser_test = subparsers.add_parser("test", help="Load a model for testing.")
    parser_test._action_groups.pop()
    parser_test_required_args = parser_test.add_argument_group("Required arguments")
    parser_test_optional_args = parser_test.add_argument_group("Optional arguments")

    parser_test_required_args.add_argument("--validation-file", "-v", required=True,
                                            help="Path to the SVM file containing the training data.",
                                            dest="testing_file_path")

    parser_test_required_args.add_argument("--model", "-m", required=True,
                                           help="The name of the file from which to load the model.", dest="model_path")

    parser_test_required_args.add_argument("--results-dir", "-r", required=True,
                                            help="Path to the directory to store results.",
                                            dest="results_dir")

    parser_test_optional_args.add_argument("--threshold", required=False, type=float, default="0.5", help="Threshold for valid track", dest="threshold")
    parser_test_optional_args.add_argument("--batch-size", "-b", required=False, type=int, default="8",
                                           help="Size of the evaluation batch.", dest="batch_size")

    #parser_test_optional_args.add_argument("--px", required=False, type=int, default=0,
    #                                        help="X axis zero padding (width). Both the left and right sides of the input will have this padding applied.", dest="padding_x")

    #parser_test_optional_args.add_argument("--py", required=False, type=int, default=0,
    #                                        help="Y axis zero padding (height). Both the top and bottom sides of the input will have this padding applied.", dest="padding_y")

    # Create prediction subprogram parser
    parser_predict = subparsers.add_parser("predict", help="Load a model and use it for predictions.")
    parser_predict._action_groups.pop()
    parser_predict_required_args = parser_predict.add_argument_group("Required arguments")
    parser_predict_optional_args = parser_predict.add_argument_group("Optional arguments")

    parser_predict_required_args.add_argument("--prediction-file", "-p", required=True,
                                              help="Path to the SVM file containing the prediction data.",
                                              dest="prediction_file_path")

    parser_predict_required_args.add_argument("--model", "-m", required=True,
                                              help="The name of the file from which to load the model.",
                                              dest="model_path")

    parser_predict_required_args.add_argument("--results-dir", "-r", required=True,
                                              help="Path to the directory to store results.",
                                              dest="results_dir")

    parser_predict_optional_args.add_argument("--batch-size", "-b", required=False, type=int, default="32",
                                              help="Size of the prediction batch.", dest="prediction_batch_size")
    parser_predict_optional_args.add_argument("--threshold", required=False, type=float, default="0.5", help="Threshold for valid track", dest="threshold")

    #parser_predict_optional_args.add_argument("--px", required=False, type=int, default=0,
    #                                       help="X axis zero padding (width). Both the left and right sides of the input will have this padding applied.", dest="padding_x")

    #parser_predict_optional_args.add_argument("--py", required=False, type=int, default=0,
    #                                       help="Y axis zero padding (height). Both the top and bottom sides of the input will have this padding applied.", dest="padding_y")


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

    if input_type == "train":
        # Read training and testing data
        X_train, y_train, train_tracks_per_event = read_svm_to_X_Y_datasets(args.training_file_path, 4032)
        X_test, y_test, test_tracks_per_event = read_svm_to_X_Y_datasets(args.testing_file_path, 4032)

        return {
            "training": {"data": X_train, "labels": y_train, "tracks_per_event": train_tracks_per_event ,"epochs": args.epochs, "batch_size": args.batch_size},
            "testing": {"data": X_test, "labels": y_test, "tracks_per_event": test_tracks_per_event , "batch_size": args.batch_size},
            "configuration": {"layers": 36, "sensors_per_layer": 112}
        }

    elif input_type == "test":
        # Read testing data
        X_test, y_test, test_tracks_per_event = read_svm_to_X_Y_datasets(args.testing_file_path, 4032)

        return {
            "testing": {"data": X_test, "labels": y_test, "tracks_per_event": test_tracks_per_event , "batch_size": args.batch_size, "threshold": args.threshold},
            "configuration": {"layers": 36, "sensors_per_layer": 112}
        }

    elif input_type == "predict":
        X_test, y_test, predict_tracks_per_event = read_svm_to_X_Y_datasets(args.prediction_file_path, 4032)

        return {
            "prediction": {"data": X_test, "labels": y_test, "threshold": args.threshold},
            "configuration": {"layers": 36, "sensors_per_layer": 112}
        }

    else:
        print(colored("Error: Wrong input type.", "red"))
        quit()


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

    if not args.nn or args.nn == "0":
        print("Training model 0")
        from models.CnnDenoisingModel0 import CnnDenoisingModel0 as CnnDenoisingModel
    elif args.nn == "0a":
        print("Training model 0a")
        from models.CnnDenoisingModel0a import CnnDenoisingModel0a as CnnDenoisingModel
    elif args.nn == "0b":
        print("Training model 0b")
        from models.CnnDenoisingModel0b import CnnDenoisingModel0b as CnnDenoisingModel
    elif args.nn == "0c":
        print("Training model 0c")
        from models.CnnDenoisingModel0c import CnnDenoisingModel0c as CnnDenoisingModel
    elif args.nn == "0d":
        print("Training model 0d")
        from models.CnnDenoisingModel0d import CnnDenoisingModel0d as CnnDenoisingModel
    elif args.nn == "0e":
        print("Training model 0e")
        from models.CnnDenoisingModel0e import CnnDenoisingModel0e as CnnDenoisingModel
    elif args.nn == "0f":
        print("Training model 0f")
        from models.CnnDenoisingModel0f import CnnDenoisingModel0f as CnnDenoisingModel
    elif args.nn == "0g":
        print("Training model 0g")
        from models.CnnDenoisingModel0g import CnnDenoisingModel0g as CnnDenoisingModel
        # Add padding
        input_dict["configuration"] = {"layers": 36, "sensors_per_layer": 112, "padding_x": 1, "padding_y": 0}
    elif args.nn == "1":
        print("Training model 1")
        from models.CnnDenoisingModel1 import CnnDenoisingModel1 as CnnDenoisingModel
    elif args.nn == "2":
        print("Training model 2")
        from models.CnnDenoisingModel2 import CnnDenoisingModel2 as CnnDenoisingModel

    model = CnnDenoisingModel(input_dict=input_dict)
    model.build_new_model()

    train_dict = model.preprocess_input(input_dict)

    training_metrics = model.train(train_dict)
    training_metrics["results_dir"] = results_dir

    model.save_model(results_dir+'cnn_autoenc')
    print_training_report(training_metrics)
    plot_train_val_graph(training_metrics)
    print(colored(f'\nSaving training results to {results_dir}\n', "green"))


def test_model(args):
    from models.CnnDenoisingModelBase import CnnDenoisingModelBase

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

    model = CnnDenoisingModelBase(input_dict=input_dict)
    model.load_model(args.model_path)

    test_dict = model.preprocess_input(input_dict)

    testing_metrics = model.test(test_dict)
    testing_metrics["input"] = input_dict["testing"]["data"]
    threshold = input_dict["testing"]["threshold"]
    testing_metrics["threshold"] = threshold
    testing_metrics["results_dir"] = results_dir
    testing_metrics["tracks_per_sample"] = input_dict["testing"]["tracks_per_event"]

    if model.use_padding:
        testing_metrics["truth"] = input_dict["testing"]["labels"]
        testing_metrics["predictions"] = model.remove_padding(testing_metrics["predictions"])

    print_testing_report(testing_metrics)
    plot_accuracy_histogram(testing_metrics)
    plot_random_predicted_events(results_dir, input_dict["testing"]["labels"], input_dict["testing"]["data"], testing_metrics["predictions"], 6, threshold)
    print(colored(f'\nSaving testing results to {results_dir}\n', "green"))


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

    model = CnnDenoisingModelBase(input_dict=input_dict)
    model.load_model(args.model_path)

    predict_dict = model.preprocess_input(input_dict)

    predict_metrics = model.predict(predict_dict)
    # predict_metrics["results_dir"] = results_dir

    if model.use_padding:
        predict_metrics["predictions"] = model.remove_padding(predict_metrics["predictions"])

    denoised = predict_metrics["predictions"]
    raw = input_dict["prediction"]["data"]
    clean = input_dict["prediction"]["labels"]

    write_raw_clean_denoised_to_svm(results_dir+"prediction.lsvm", raw, clean, denoised, 4032)

    # predict_metrics["input_data"] = input_dict["prediction"]["data"]
    # predict_metrics["num_planes"] = input_dict["configuration"]["total_planes"]
    # predict_metrics["path_to_df"] = input_dict["prediction"]["input"]
    # predict_metrics["quantization"] = input_dict["prediction"]["quantization"]
    # save_prediction_results(predict_metrics)


def plot_random_predicted_events(results_dir, ground_truth, noisy, prediction, num_random_events, threshold, seed = 22):
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
        plt.imsave(results_dir+'denoised'+str(i)+'.png', (prediction[index].reshape(img_shape)>threshold).astype(int))


def plot_accuracy_histogram(testing_metrics):
    """
    Plots the accuracy histograms, hits and noise

    Args:
        training_metrics (dictionary): Dictionary outputted by the training function
    """
    threshold = testing_metrics["threshold"]
    ground_truth = testing_metrics["truth"]
    ground_truth = ground_truth.reshape((-1, ground_truth.shape[-3] * ground_truth.shape[-2]))
    predictions = testing_metrics["predictions"]
    predictions = predictions.reshape((-1, predictions.shape[-3] * predictions.shape[-2]))
    raw_input = testing_metrics["input"]
    raw_input = raw_input.reshape((-1, raw_input.shape[-3] * raw_input.shape[-2]))

    results_dir = testing_metrics["results_dir"]

    hits_stats = hm.plot_hits(results_dir+'hits_histogram.png', predictions, ground_truth, results_dir, threshold)
    noise_stats = hm.plot_noise(results_dir+'noise_histogram.png', predictions, ground_truth, results_dir, threshold)
    noise_reduction_stats = hm.plot_noise_reduction(results_dir+'noise_reduction_histogram.png', predictions, raw_input, ground_truth, results_dir, threshold)
    hits_per_segment = hm.plot_hits_per_segment(results_dir+'hits_per_segment.png', predictions, ground_truth, testing_metrics["tracks_per_sample"], threshold)
    
    
    cases = hits_per_segment["num"]
    hits_max = hits_stats["max"]
    hits_min = hits_stats["min"]
    hits_mean = hits_stats["mean"]
    hits_rms = hits_stats["rms"]

    print(f'{colored("Total number of tracks:", "blue")} {int(cases)}')
    print(f'{colored("Hits Minimum value(%):", "blue")} {hits_min}')
    print(f'{colored("Hits Maximum value(%)::", "blue")} {hits_max}')
    print(f'{colored("Hits Mean value(%)::", "blue")} {hits_mean}')
    print(f'{colored("Hits RMS(%)::", "blue")} {hits_rms}')

    noise_max = noise_stats["max"]
    noise_min = noise_stats["min"]
    noise_mean = noise_stats["mean"]
    noise_rms = noise_stats["rms"]

    print(f'{colored("Noise Minimum value(%):", "blue")} {noise_min}')
    print(f'{colored("Noise Maximum value(%)::", "blue")} {noise_max}')
    print(f'{colored("Noise Mean value(%)::", "blue")} {noise_mean}')
    print(f'{colored("Noise RMS value(%)::", "blue")} {noise_rms}')


    noise_reduction_max = noise_reduction_stats["max"]
    noise_reduction_min = noise_reduction_stats["min"]
    noise_reduction_mean = noise_reduction_stats["mean"]
    noise_reduction_rms = noise_reduction_stats["rms"]
    init_noise_mean = noise_reduction_stats["init_noise_mean"]
    init_noise_rms  = noise_reduction_stats["init_noise_rms"]
    rec_noise_mean  = noise_reduction_stats["rec_noise_mean"]
    rec_noise_rms   = noise_reduction_stats["rec_noise_rms"]

    print(f'{colored("Noise Reduction Minimum value(%):", "blue")} {noise_reduction_min}')
    print(f'{colored("Noise Reduction Maximum value(%)::", "blue")} {noise_reduction_max}')
    print(f'{colored("Noise Reduction Mean value(%)::", "blue")} {noise_reduction_mean}')
    print(f'{colored("Noise Reduction RMS value(%)::", "blue")} {noise_reduction_rms}')


    print(f'{colored("init_noise_mean(%)::", "blue")} {init_noise_mean}')
    print(f'{colored("init_noise_rms(%)::", "blue")} {init_noise_rms}')
    print(f'{colored("rec_noise_mean(%)::", "blue")} {rec_noise_mean}')
    print(f'{colored("rec_noise_rms(%)::", "blue")} {rec_noise_rms}')

    reconstruct_6_super_layers = hits_per_segment["valid-6"]
    reconstruct_5_super_layers = hits_per_segment["valid-5"]
    reconstruct_4_super_layers = hits_per_segment["valid-4"]

    # print(f'{colored('Reconstructed from 6 superlayers(%): ',"blue")}  {(reconstruct_6_super_layers / cases) * 100)}' )
    # print(f'{colored('Reconstructed from 5 superlayers(%): ',"blue")}  {(reconstruct_5_super_layers / cases) * 100)}' )
    # print(f'{colored('Reconstructed from 4 superlayers(%): ',"blue")}  {(reconstruct_4_super_layers / cases) * 100)}' )


    with open(results_dir + 'testing_report.txt','a+') as f:
        f.write('Total number of cases: '+ str(cases) + '\n')
        f.write('Hits Minimum value(%): ' + str(hits_min) + '\n')
        f.write('Hits Maximum value(%): ' + str(hits_max) + '\n')
        f.write('Hits Mean value(%): ' + str(hits_mean) + '\n')
        f.write('Hits RMS value(%): ' + str(hits_rms) + '\n')
        f.write('Noise Minimum value(%): ' + str(noise_min) + '\n')
        f.write('Noise Maximum value(%): ' + str(noise_max) + '\n')
        f.write('Noise Mean value(%): ' + str(noise_mean) + '\n')
        f.write('Noise RMS value(%): ' + str(noise_rms) + '\n')
        f.write('Noise Reduction Minimum value(%): ' + str(noise_reduction_min) + '\n')
        f.write('Noise Reduction Maximum value(%): ' + str(noise_reduction_max) + '\n')
        f.write('Noise Reduction Mean value(%): ' + str(noise_reduction_mean) + '\n')
        f.write('Noise Reduction RMS value(%): ' + str(noise_reduction_rms) + '\n')

        f.write('Reconstructed from 6 superlayers: ' + str(reconstruct_6_super_layers) + '\n')
        f.write('Reconstructed from 5 superlayers: ' + str(reconstruct_5_super_layers) + '\n')
        f.write('Reconstructed from 4 superlayers: ' + str(reconstruct_4_super_layers) + '\n')

        f.write('Reconstructed from 6 superlayers(%): ' + str((reconstruct_6_super_layers / cases) * 100) + '\n')
        f.write('Reconstructed from 5 superlayers(%): ' + str((reconstruct_5_super_layers / cases) * 100) + '\n')
        f.write('Reconstructed from 4 superlayers(%): ' + str((reconstruct_4_super_layers / cases) * 100) + '\n')

        f.write('init_noise_mean(%): ' + str(init_noise_mean) + '\n')
        f.write('init_noise_rms(%): ' + str(init_noise_rms) + '\n')
        f.write('rec_noise_mean(%): ' + str(rec_noise_mean) + '\n')
        f.write('rec_noise_rms(%): ' + str(rec_noise_rms) + '\n')


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

    with open(training_results_dir + 'training_report.txt','w+') as f:
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

    with open(testing_results_dir + 'testing_report.txt','w+') as f:
        f.write("Testing loss: " + str(testing_loss) +'\n')
        f.write("Testing prediction time: " + str(testing_prediction_time) +'\n')

    print("\nTesting Report")
    print("================================")
    print(f'{colored("Testing loss:", "blue")} {testing_loss}')
    print(f'{colored("Testing prediction time:", "blue")} {testing_prediction_time}s')


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
