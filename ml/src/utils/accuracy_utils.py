#############################################################################
# This file contains a series of utility functions for computing
# the accuracy of our machine learning models in regards to the specific
# problem.
#############################################################################

import numpy as np
import math
from sklearn.datasets import dump_svmlight_file
def get_accuracy_A1(confusion_matrix, total_samples):
    """
    Computes the A1 accuracy given the confusion matrix and the total number
    of samples.

    The A1 accuracy is defined as the perecentage of samples where the valid track
    was correctly detected.
    
    Args:
        confusion_matrix: The confusion matrix
        total_samples (int): The total number of samples
    
    Returns:
        The A1 accuracy
    """

    A1_count = 0
    for i in range(1, confusion_matrix.shape[1]):
        A1_count += confusion_matrix[i,i]
    print(A1_count)
    print(total_samples)
    return A1_count / total_samples

def get_accuracy_Ac(confusion_matrix, predictions, segmented_test_labels):
    """
    Computes the Ac accuracy given the confusion matrix, the predictions from
    the machine learning model, and the test dataset labels in segmented format.

    The Ac accuracy is defined as the percentage of predicted valid samples that contain
    mispredicted rows.
    
    Args:
        confusion_matrix: The confusion matrix
        predictions: Predictions outputted by an ML model
        segmented_test_labels: Segmented array of the test labels
    
    Returns:
        The Ac accuracy
    """

    predicted_samples_with_errors = 0

    iter_index = 0

    for sample in segmented_test_labels:
        first_label_index = iter_index
        found_invalid = False

        if predictions[first_label_index] == 0:
            iter_index += sample.size
            continue

        for label in np.nditer(sample):
            if iter_index == first_label_index:
                iter_index += 1
                continue

            if not found_invalid and predictions[iter_index] != 0:
                predicted_samples_with_errors += 1
                found_invalid = True

            iter_index += 1

    A1_count = 0
    for i in range(1, confusion_matrix.shape[1]):
        A1_count += confusion_matrix[i,i]

    return predicted_samples_with_errors / A1_count

def get_accuracy_Ac_g(confusion_matrix, total_rows, total_samples):
    """
    Computes the Ac accuracy given the confusion matrix and the total number
    of samples.

    The Ac accuracy is defined as the global percentage of rows that were mispredicted
    as valid ones.
    
    Args:
        confusion_matrix: The confusion matrix
        total_rows (int): The total number of rows in the dataset
        total_samples (int): The total number of samples in the dataset
    
    Returns:
        The Ac_g accuracy
    """

    return confusion_matrix[0,1] / (total_rows - total_samples)

def get_accuracy_new_Ah(predicted_probabilities, segmented_test_labels, segmented_features):
    
    max_index = 0
    iter_index = 0
    num_Ah = 0
    segmented_false_positives_probabilities = []
    segmented_false_positives_features = []
    segmented_false_positives_truth_features = []
    segmented_false_positives_truth_probabilities = []

    for sample, y_sample in zip(segmented_features, segmented_test_labels):
        first_row_index = iter_index
        max_index = first_row_index
        count = True
        argmax = np.argmax(predicted_probabilities[first_row_index])
        sample = np.array(sample)
        for row in sample:
            # print(row[0].shape)
            # print(row,  predicted_probabilities[iter_index])
            if argmax != int(y_sample[0]):
                count = False 
            elif predicted_probabilities[first_row_index][argmax] < predicted_probabilities[iter_index][argmax]:

                if first_row_index != iter_index:
                    segmented_false_positives_truth_features.append(sample[0])
                    segmented_false_positives_truth_probabilities.append(predicted_probabilities[first_row_index])
                    segmented_false_positives_features.append(row)
                    segmented_false_positives_probabilities.append(predicted_probabilities[iter_index])
                    for i in range(0,6):
                        if math.isclose(row[i],sample[0,i]):
                            count = False
      
            iter_index += 1

        if count == True:
            num_Ah +=1

    total_samples = len(segmented_features)
    # print(segmented_false_positives_features)
    # print(np.array(segmented_false_positives_features))
    # print(np.array(segmented_false_positives_probabilities))
    # dump_svmlight_file(segmented_false_positives_features, segmented_false_positives_probabilities, 'false_positives_no_line.lsvm', zero_based = False, multilabel=True)
    with open('false_positives_no_line.lsvm', 'w+') as f:
        for entry in  zip(segmented_false_positives_truth_features, segmented_false_positives_truth_probabilities, segmented_false_positives_features, segmented_false_positives_probabilities):
            # f.write('{} {} {} 1:{} 2:{} 3:{} 4:{} 5:{} 6:{} \n{} {} {} 1:{} 2:{} 3:{} 4:{} 5:{} 6:{}\n================================= \n'.format(entry[1][0], entry[1][1], entry[1][2], entry[0][0], entry[0][1], entry[0][2], entry[0][3], entry[0][4], entry[0][5], entry[3][0], entry[3][1], entry[3][2], entry[2][0], entry[2][1], entry[2][2], entry[2][3], entry[2][4], entry[2][5]))
            f.write('{} {} {} 1:{} 2:{} 3:{} 4:{} 5:{} 6:{} \n{} {} {} 1:{} 2:{} 3:{} 4:{} 5:{} 6:{}\n'.format(entry[1][0], entry[1][1], entry[1][2], entry[0][0], entry[0][1], entry[0][2], entry[0][3], entry[0][4], entry[0][5], entry[3][0], entry[3][1], entry[3][2], entry[2][0], entry[2][1], entry[2][2], entry[2][3], entry[2][4], entry[2][5]))
    return num_Ah / total_samples



def get_accuracy_Ah(predicted_probabilities, segmented_test_labels):
    """
    Computes the Ah accuracy given an array of the predicted probabilities
    for each row in the dataset as well as the test labels in segmented
    format (using svm_utils).

    The Ah accuracy is defined as the perecentage of predictions where the
    label correctly predicted as 1 has the highest probability of all other labels
    being a 1 in a sample.
    
    Args:
        predicted_probabilities: Flat array of predicted probabilities in tuple format
        (0: probability of being 0, 1: probability of being 1)
        segmented_test_labels: Segmented array of the test labels
    
    Returns:
        The Ah accuracy
    """

    correct_predictions = 0

    max_index = 0
    iter_index = 0

    for sample in segmented_test_labels:
        first_label_index = iter_index
        max_index = first_label_index
        count = True
        argmax = np.argmax(predicted_probabilities[first_label_index])
        for label in np.nditer(sample):
            if argmax != int(sample[0]):
                # print(np.argmax(predicted_probabilities[first_label_index]), int(sample[0]))
                count = False
            elif predicted_probabilities[first_label_index][argmax] < predicted_probabilities[iter_index][argmax]:
                if first_label_index != iter_index:
                    count = False
            iter_index += 1

        # If the highest probability of 1 has the index of the first label
        # in the sample, then it is correct
        if count == True:
            correct_predictions += 1

    total_samples = len(segmented_test_labels)

    return correct_predictions / total_samples

def get_accuracy_Af(confusion_matrix, total_samples):
    """
    Computes the Af accuracy given the confusion matrix and the total number
    of samples.

    The Af accuracy is defined as the perecentage of samples where the valid track
    was not detected.
    
    Args:
        confusion_matrix: The confusion matrix
        total_samples (int): The total number of samples
    
    Returns:
        The Af accuracy
    """
    Af_count = 0
    for i in range(1, confusion_matrix.shape[1]):
        for j in range(0, confusion_matrix.shape[1]):
            if i == j:
                continue
            Af_count += confusion_matrix[i,j]

    return Af_count / total_samples