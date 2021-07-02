import csv
import matplotlib.pyplot as plt
import numpy as np


all_hits = {}
all_tracks = {}
all_init_noise = []
all_rec_noise = {}
markers = ["o", "s", "D", "^", "v", "<"]
for b in [45, 50, 55, 90, 100, 110]: 
    new = True
    if b not in all_hits:
        all_hits[b] = []
        # all_init_noise[b] = []
        all_rec_noise[b] = []
    thresholds = []    
    for t in ["0.05", "0.10", "0.15", "0.20", "0.25","0.30","0.35","0.40","0.45","0.5"]:
        thresholds.append(t)
        curr_dir = str(b)+"/"+t+"/all_hits_data.csv" 
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
            all_hits[b].append(np.array(hits).mean())
            if new:
                all_init_noise.append(np.array(init_noise).mean())
                new = False
            all_rec_noise[b].append(np.array(rec_noise).mean())
        with open(str(b)+"/"+t+"/testing_report.txt",'r' ) as test_file:
            tot = 0.0
            if t not in all_tracks:
                    all_tracks[t] = []
            for line in test_file:
                if "Reconstructed from 6 superlayers(%):" in line:
                    tot +=( float(line.split(": ")[1])/100)
                elif "Reconstructed from 5 superlayers(%):" in line:
                    tot +=( float(line.split(": ")[1])/100)
                elif "Reconstructed from 4 superlayers(%):" in line:
                    tot +=(  float(line.split(": ")[1])/100)
            all_tracks[t].append(tot)


for b ,m in zip([45, 50, 55, 90, 100, 110], markers):
    plt.scatter(thresholds, all_rec_noise[b], edgecolors='k', marker=m, label=str(b)+'nA')
plt.xlabel("Threshold")
plt.ylabel("Noise Fraction")
plt.legend(loc="best")
plt.savefig('../noise_fraction_vs_threshold.png')
# plt.show()
plt.figure()


# for b ,m in zip([45, 50, 55, 90, 100, 110], markers):
plt.scatter([45, 50, 55, 90, 100, 110], all_init_noise, edgecolors='k', marker='s')
plt.xlabel("Beam Current (nA)")
plt.ylabel("All Hits/Track Hits")
plt.ylim([1,8])
# plt.show()
plt.savefig('../all_hits_vs_track_hits.png')
plt.figure()

for b ,m in zip([45, 50, 55, 90, 100, 110], markers):
    plt.scatter(thresholds, all_hits[b], edgecolors='k', marker=m, label=str(b)+'nA')
plt.xlabel("Threshold")
plt.ylabel("Hit reconstruction Efficiency")
plt.ylim([0.75,1])
plt.legend(loc="best")
# plt.show()
plt.savefig('../hit_rec_vs_threshold.png')
plt.figure()

for t ,m in zip(["0.05", "0.10", "0.20", "0.30", "0.40","0.5"], markers):
    plt.scatter([45, 50, 55, 90, 100, 110], all_tracks[t], edgecolors='k', marker=m, label=str(t))
plt.xlabel("Beam Current (nA)")
plt.ylabel("Track reconstruction Efficiency")
plt.ylim([0.75,1])
plt.legend(loc="best")
# plt.show()
plt.savefig('../track_rec_vs_threshold.png')
plt.figure()
