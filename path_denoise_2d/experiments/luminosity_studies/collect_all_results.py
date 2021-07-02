import csv
import matplotlib.pyplot as plt
import numpy as np

def rmsd(y):
    y_mean = np.mean(y)
    return np.sqrt(np.mean(((y-y_mean)**2), axis=0))

def error_bar_charts(x, m, rms, x_label, y_label, clr, lab):
    # plt.figure()
    plt.ylim([0,1])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.errorbar(x, m, yerr=rms, fmt='o', color=clr, label=lab)

models = []    
all_hits = []
all_hits_err = []
all_init_noise = []
all_rec_noise = []
all_rec_noise_err = []
for p in ["45", "50", "55", "90","100","110"]:
    models.append(p)
    curr_dir = p+"/all_hits_data.csv" 
    line = 0
    with open(curr_dir, 'r') as file:
        reader = csv.reader(file)
        hits = []
        init_noise = []
        rec_noise = []
        line = 0
        for row in reader:
            if line == 0:
                line += 1
                continue
            line += 1
            hits.append(float(row[6]))
            init_noise.append(float(row[8]))
            rec_noise.append(float(row[9]))
        hits = np.array(hits)
        rec_noise = np.array(rec_noise)
        all_hits_err.append(rmsd(hits))
        all_rec_noise_err.append(rmsd(rec_noise))
        all_hits.append(np.array(hits).mean())
        all_init_noise.append(np.array(init_noise).mean())
        all_rec_noise.append(np.array(rec_noise).mean())
error_bar_charts(models, all_hits, all_hits_err, "Beam Current(nA)", "Efficiency","b", "Hits fraction")
error_bar_charts(models, all_rec_noise, all_rec_noise_err, "Beam Current(nA)", "Efficiency","r", "Noise fraction")
# plt.show()
plt.legend(loc="best")
plt.savefig("../luminosity_studies_eff.png")
plt.figure()
