import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
import csv

def error_bar_charts(filename, x_ticks, m, rmsd, x_label, y_label, ):
    ''' Plot with errorbars. X values can either be strings or numbers
    Parameters:
    filename (string): The file to store the resulting figure
    x_ticks (list of strings): The labels for each x value
    m (1D array): The mean value for the dataset
    rmsd (1D array): The rmsd value for the dataset
    x_label (string): The label of the x axis
    y_label (string): The label of the y axis

    '''

    plt.ylim([0,120])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(np.arange(len(x_ticks)), x_ticks)
    plt.errorbar(np.arange(len(x_ticks)), m, yerr=rmsd, fmt='o')

    plt.savefig(filename)
    # plt.show() # Uncomment to show figure after plotting


# Example usage

def rmsd(y):
    y_mean = np.mean(y)
    return np.sqrt(np.mean(((y-y_mean)**2), axis=0))

y1 = np.random.random((50, 10)) * 100 

y1_mean = np.mean(y1, axis=0)
y1_error_root = rmsd(y1)


x_ticks = ['45', '50', '55', '60', '65', '70', '80', '90', '100', '110']



error_bar_charts("error_bar.pdf", x_ticks, y1_mean, y1_error_root, "x_label", "y_label")