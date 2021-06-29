import csv
import matplotlib.pyplot as plt
import numpy as np


models = []    
all_hits = []
all_init_noise = []
all_rec_noise = []
for p in ["0", "0a", "0b", "0c","0d","0e","0f","0g","1", "2"]:
    models.append(p)
    curr_dir = p+"/test/all_hits_data.csv" 
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
        all_hits.append(np.array(hits).mean())
        all_init_noise.append(np.array(init_noise).mean())
        all_rec_noise.append(np.array(rec_noise).mean())
plt.scatter(models, all_hits, marker="s", edgecolors='k',label="Track hits fraction")
plt.scatter(models, all_rec_noise, edgecolors='k',label="Noise fraction")
plt.xlabel("Model")
plt.ylabel("Efficiency")
plt.ylim([0,1])
plt.legend(loc="best")
# plt.show()
plt.savefig('../model_studies_eff.png')
plt.figure()
plt.scatter(models, np.array(all_init_noise)/np.array(all_rec_noise), marker="^", edgecolors='g')
plt.xlabel("Model")
plt.ylabel("De-Noising Power")
plt.ylim([0,30])
# plt.show()
plt.savefig('../smodel_studies_denoise_power.png')