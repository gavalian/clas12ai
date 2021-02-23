import csv
import numpy as np
import matplotlib.pyplot as plt
def bar_charts(bar_6, bar_5, bar_4, x_label, y_label, filename):
    # Numbers of pairs of bars you want
    N = 6

    # Data on X-axis

    ind = np.arange(N)

    plt.figure()

    # Width of a bar 
    width = 0.1       

    # Plotting
    plt.bar(ind, bar_6 + bar_5 + bar_4 , width, label='Total')
    plt.bar(ind + width, bar_6 , width, label='6-segments')
    plt.bar(ind + width + width, bar_5, width, label='5-segments')
    plt.bar(ind + width + width + width, bar_4, width, label='4-segments')

    plt.xlabel('Luminosity (nA)')
    plt.ylabel('Tracks reconstructed (%)')

    # xticks()
    # First argument - A list of positions at which ticks should be placed
    # Second argument -  A list of labels to place at the given locations
    plt.xticks(ind + width + width / 2 , ('45', '50', '55', '90', '100', '110'))
    plt.yticks(np.arange(0, 110, 10))
    plt.ylim([0,100])

    # Position legend
    plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=4)
    plt.tight_layout()
    plt.savefig(filename)

# Example
# n = "/path/to/csv/file.csv"
# output = "/path/to/destination/file.pdf"
# with open(n) as file:
#    reader = csv.reader(file)
#    i = 0
#    segs = [[], [], []]
#    next(reader)

#    for row in reader:
#        for col in [3, 2, 1]:
#            segs[-col].append(float(row[-col])/float(row[-4])*100)
#            i += 1

#    x_label = "Luminosity (nA)"
#    y_label = "Reconstructions"
#    bar_charts(np.array(segs[-3]), np.array(segs[-2]), np.array(segs[-1]), x_label, y_label, output)




