import matplotlib.pyplot as plt
import numpy as np

def bar_charts(bar_6, bar_5, bar_4, x_label, y_label):

    # Numbers of pairs of bars you want
    N = 6

    # Data on X-axis

    ind = np.arange(N)

    plt.figure()

    # Width of a bar 
    width = 0.1       

    # Plotting
    plt.bar(ind, bar_6 + bar_5 + bar_4 , width, edgecolor='k', label='4-segments')
    
    plt.bar(ind + width, bar_5 + bar_6, width, edgecolor='k', label='5-segments')
    plt.bar(ind + width + width, bar_6 , width, edgecolor='k', label='6-segments')    
    #plt.bar(ind + width + width + width, bar_4, width, label='4-segments')

    plt.xlabel('Luminosity (nA)')
    plt.ylabel('Tracks reconstructed (%)')

    # First argument - A list of positions at which ticks should be placed
    # Second argument -  A list of labels to place at the given locations
    plt.xticks(ind + width + width / 2 , ('45', '50', '55', '90', '100', '110'))
    plt.yticks(np.arange(0, 1.1, 0.10))
    plt.ylim([0,1])

    # Position legend
    plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=4)
    plt.tight_layout()
    # plt.savefig(filename)



models = []    
all_hits = []
all_hits_err = []
all_init_noise = []
all_rec_noise = []
all_rec_noise_err = []
all_rec_6 = []
all_rec_5 = []
all_rec_4 = []
for p in ["45", "50", "55", "90","100","110"]:
    models.append(p)
    curr_dir = p+"/testing_report.txt" 
    line = 0
    with open(curr_dir, 'r') as file:
        for line in file:
            if "Reconstructed from 6 superlayers(%):" in line:
                all_rec_6.append( float(line.split(": ")[1])/100)
            elif "Reconstructed from 5 superlayers(%):" in line:
                all_rec_5.append( float(line.split(": ")[1])/100)
            elif "Reconstructed from 4 superlayers(%):" in line:
                all_rec_4.append(  float(line.split(": ")[1])/100)
bar_charts(np.array(all_rec_6), np.array(all_rec_5), np.array(all_rec_4), None, None)
# plt.show()
plt.savefig('../segments_reconstruction.png')
plt.figure()