#!/usr/bin/env python3

import sys
sys.path.append("..")

import argparse
import numpy as np
from termcolor import colored

from common.svm_utils import read_concat_svm_files, filter_rows_with_label
from models.GruModel import GruModel

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
    parser = argparse.ArgumentParser(description="CRTC-JLab Machine Learning CLI")
    subparsers = parser.add_subparsers()
    subparsers.required = True
    subparsers.dest = "subprogram"

    # Create training subprogram parser
    parser_train = subparsers.add_parser("train", help="Train a model, perform testing, and serialize it.")
    parser_train._action_groups.pop()
    parser_train_required_args = parser_train.add_argument_group("Required arguments")
    parser_train_optional_args = parser_train.add_argument_group("Optional arguments")
    parser_train_required_args.add_argument("--training-file", "-t", required=True, help="Path to the file containing the training data.", dest="training_file_path")
    parser_train_required_args.add_argument("--testing-file", "-e", required=True, help="Path to the file containing the testing data.", dest="testing_file_path")
    parser_train_required_args.add_argument("--out-model", "-m", required=True, help="Name of the file in which to save the model.", dest="output_model_path")
    parser_train_optional_args.add_argument("--epochs", required=False, type=int, default="20", help="How many training epochs to go through.", dest="epochs")
    parser_train_optional_args.add_argument("--batch-size", required=False, type=int, default="16", help="Size of each batch.", dest="batch_size")

    # Create evaluation subprogram parser
    parser_test = subparsers.add_parser("test", help="Load a model for testing.")
    parser_test._action_groups.pop()
    parser_test_required_args = parser_test.add_argument_group("Required arguments")
    parser_test_optional_args = parser_test.add_argument_group("Optional arguments")
    parser_test_required_args.add_argument("--testing-file", "-e", required=True, help="Path to the file containing the testing data.", dest="testing_file_path")
    parser_test_required_args.add_argument("--model", "-m", required=True, help="The name of the file from which to load the model.", dest="model_path")
    parser_test_optional_args.add_argument("--batch-size", required=False, type=int, default="16", help="Size of each batch.", dest="batch_size")

    # Create prediction subprogram parser
    parser_predict = subparsers.add_parser("predict", help="Load a model and use it for predictions.")
    parser_predict._action_groups.pop()
    parser_predict_required_args = parser_predict.add_argument_group("Required arguments")
    parser_predict_optional_args = parser_predict.add_argument_group("Optional arguments")
    parser_predict_required_args.add_argument("--prediction-file", "-p", required=True, help="Path to the file containing the prediction data.", dest="prediction_file_path")
    parser_predict_required_args.add_argument("--model", "-m", required=True, help="The name of the file from which to load the model.", dest="model_path")
    # parser_predict_required_args.add_argument("--model-type", choices=["cnn", "mlp", "et"], required=True, help="The type of the model to load.", dest="model_type")
    parser_predict_optional_args.add_argument("--batchSize", required=False, type=int, default="32", help="Size of the prediction batch.", dest="prediction_batch_size")
    parser_predict_optional_args.add_argument("--c-lib", required=False, default="libc_reader.so", help="Path to the C library reader interface", dest="c_library")

    return parser.parse_args()


def read_input_data(input_type, args) -> dict:
    """
    Reads all of the SVM files in the training and testing directories,
    concatenates them, and creates a dictionary that references them.

    Args:
        input_type: Type of reading and processing of the input for training, evaluation, or prediction.
        args: The object that contains all the parsed CLI arguments.

    Returns:
        A dictionary containing the read and processed input data. This includes the features and targets when appropriate.
    """

    if input_type == "train":
        # Read training and testing data
        X_train, y_train = read_concat_svm_files([args.training_file_path], 36)
        X_test, y_test = read_concat_svm_files([args.testing_file_path], 36)

        X_train, y_train = filter_rows_with_label(X_train, y_train, 1)
        X_test, y_test = filter_rows_with_label(X_test, y_test, 1)

        return {
            "training": {
                "data": X_train,
                "labels": y_train,
                "epochs": args.epochs,
                "batch_size": args.batch_size
            },
            "testing": {
                "data": X_test,
                "labels": y_test,
                "batch_size": args.batch_size
            }
        }

    elif input_type == "test":
        # Read testing data
        X_test, y_test = read_concat_svm_files([args.testing_file_path], 36)

        X_test, y_test = filter_rows_with_label(X_test, y_test, 1)

        return {
            "testing": {
                "data": X_test,
                "labels": y_test,
                "batch_size": args.batch_size
            }
        }

    # elif input_type == "predict":
    #    return {"prediction": {"data": prediction_data}}

    else:
        print(colored("Error: Wrong input type.", "red"))
        quit()


def print_training_report(training_dict):
    """
    Prints a report for the training of a machine learning model.

    Args:
        training_dict: Dictionary that was outputted by the training function
    """

    training_time = training_dict["training_time"]
    training_loss = training_dict["training_loss"]

    print("\nTraining Report")
    print("================================")
    print(f'{colored("Training loss:", "blue")} {training_loss}')
    print(f'{colored("Training time:", "blue")} {training_time}s')


def print_testing_report(testing_dict):
    """
    Prints a report for the testing of a machine learning model.

    Args:
        testing_dict: Dictionary that was outputted by the testing function
    """

    testing_loss = testing_dict["testing_loss"]
    testing_prediction_time = testing_dict["testing_prediction_time"]

    print("\nTesting Report")
    print("================================")
    print(f'{colored("Testing loss:", "blue")} {testing_loss}')
    print(f'{colored("Testing prediction time:", "blue")} {testing_prediction_time}s')


def train_model(args):
    """
    Trains the model with the input data specified in the CLI arguments followed by an evaluation of the model.

    Args:
        args: The object that contains all the parsed CLI arguments.
    """

    print(colored("\nReading input data...", "green"))
    input_dict = read_input_data("train", args)

    model = GruModel(input_dict=input_dict)

    model.build_new_model()
    training_dict = model.train(input_dict)

    testing_dict = model.test(input_dict)

    print_training_report(training_dict)
    print_testing_report(testing_dict)

    model.save_model(args.output_model_path)


def test_model(args):
    """
    Evaluates the model with the input data specified in the CLI arguments.

    Args:
        args: The object that contains all the parsed CLI arguments.
    """

    print(colored("\nReading input data...", "green"))
    input_dict = read_input_data("test", args)

    model = GruModel(input_dict=input_dict)

    model.load_model(args.model_path)

    testing_dict = model.test(input_dict)

    print_testing_report(testing_dict)


def predict(args):
    """
    Uses a model for predicting values for input data read from a file
    loaded via a C library.

    Args:
        args: The object that contains all the parsed CLI arguments.
    """

    from c_py_interface.c_interface import CLibInterface

    model = GruModel()

    model.load_model(args.model_path)

    clib = CLibInterface(args.c_library)
    clib.open_file(args.prediction_file_path)

    dict_input = {}
    dict_input['prediction'] = {}

    while True:
        found = clib.read_next()
        if found.size != 24:
            break

        dict_input['prediction']['data'] = found.reshape(-1, 24, 1)
        model.preprocess_input(dict_input)
        pred = model.predict(dict_input)
        preds = pred['predictions']

        preds = preds * 112
        clib.write_roads(preds[0])


def get_subroutine(args):
    """
    Processes the CLI subprogram utilized and return the matching function for training,
    evaluation, or prediction.

    Args:
        args: The object that contains all the parsed CLI arguments

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
