from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import numpy as np
imgs_sparse, labels = load_svmlight_file("dc_denoise_one_track_fixed_5nA.lsvm")
imgs_dense = imgs_sparse.todense()
clean_data = []

for i in range(0, len(labels)):
    if labels[i] == 0:
        continue
    else:
        clean_data.append(imgs_dense[i])

clean_data = np.vstack(clean_data)

one_track_files = ["dc_denoise_one_track_fixed_45nA.lsvm", 
         "dc_denoise_one_track_fixed_50nA.lsvm",
         "dc_denoise_one_track_fixed_55nA.lsvm",
         "dc_denoise_one_track_fixed_90nA.lsvm",
         "dc_denoise_one_track_fixed_100nA.lsvm",
         "dc_denoise_one_track_fixed_110nA.lsvm"
         ]

for file in one_track_files:
    imgs_sparse, labels = load_svmlight_file(file)
    imgs_dense = imgs_sparse.todense()
    clean_track = []
    noisy_track = []

    for i in range(0, len(labels)):
        if labels[i] == 0:
            noisy_track.append(imgs_dense[i])
        else:
            clean_track.append(imgs_dense[i])

    noisy_track = np.vstack(noisy_track)

    new_labels = np.zeros((clean_data.shape[0]+ (noisy_track.shape[0])))
    new_labels[0::2] = np.ones(clean_data.shape[0])
    new_labels[1::2] = np.zeros((noisy_track.shape[0]))

    new_data = np.zeros((clean_data.shape[0] + noisy_track.shape[0], 4032))
    new_data[0::2] = clean_data
    new_data[1::2] = noisy_track
    dump_svmlight_file(new_data, new_labels, "fixed_data/"+ file, zero_based=False)


imgs_sparse, labels = load_svmlight_file("dc_denoise_two_track_fixed_5nA.lsvm")
imgs_dense = imgs_sparse.todense()
clean_data = []

for i in range(0, len(labels)):
    if labels[i] == 0:
        continue
    else:
        clean_data.append(imgs_dense[i])

clean_data = np.vstack(clean_data)

two_track_files = ["dc_denoise_two_track_fixed_45nA.lsvm", 
         "dc_denoise_two_track_fixed_50nA.lsvm",
         "dc_denoise_two_track_fixed_55nA.lsvm",
         "dc_denoise_two_track_fixed_90nA.lsvm",
         "dc_denoise_two_track_fixed_100nA.lsvm",
         "dc_denoise_two_track_fixed_110nA.lsvm"
         ]

for file in two_track_files:
    imgs_sparse, labels = load_svmlight_file(file)
    imgs_dense = imgs_sparse.todense()
    clean_track = []
    noisy_track = []

    for i in range(0, len(labels)):
        if labels[i] == 0:
            noisy_track.append(imgs_dense[i])
        else:
            clean_track.append(imgs_dense[i])

    noisy_track = np.vstack(noisy_track)

    new_labels = np.zeros((clean_data.shape[0]+ (noisy_track.shape[0])))
    new_labels[0::2] = np.ones(clean_data.shape[0]) * 2
    new_labels[1::2] = np.zeros((noisy_track.shape[0]))

    new_data = np.zeros((clean_data.shape[0] + noisy_track.shape[0], 4032))
    new_data[0::2] = clean_data
    new_data[1::2] = noisy_track
    dump_svmlight_file(new_data, new_labels, "fixed_data/"+ file, zero_based=False)