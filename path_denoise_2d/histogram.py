import numpy as np
import matplotlib.pyplot as plt
from numpy import mean
import csv
def rmsd(v, mean_value):
    v_mean = mean_value
    return np.sqrt(np.mean(((v-v_mean)**2)))

def plot_mean_with_error(path, mean, error):
    plt.axvline(mean, color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(mean+1, max_ylim*0.97, 'Mean')
    plt.errorbar(mean, max_ylim*0.75, xerr = error,capsize=3, fmt='k')
    plt.savefig(path.replace("histogram","histogram_with_mean"))

def plot_hits(path, predictions, ground_truth, dump_dir='', threshold = 0.5):
    '''
    Generates a histogram of correctly reconstructed hits

    Args:
    -----
        path: The destination to save the generated histogram
        predictions: Numpy array with the predicted hits flatenned per event
        ground_truth: Numpy array with the ground_truth hits flatenned per event
        threshold: The threshold that needs to be reached to set a value to 1
    '''

    predictions = (predictions>= threshold).astype(int)

    unique = np.zeros((ground_truth.shape[0], 1))
    i = 0
    for a1, a2 in zip(ground_truth, predictions):
        unique[i] = np.intersect1d(np.nonzero(a1)[0], np.nonzero(a2)[0]).shape[0]/ np.nonzero(a1)[0].shape[0] *100
        i += 1

    plt.figure()
    plt.xlabel('Hit accuracy %')
    plt.ylabel('Number of cases')
    plt.title('Valid Hits histogram')
    plt.xticks(np.arange(0,105,5))
    plt.hist(unique, bins=np.arange(0,105,5))
    plt.savefig(path)
    mean_value = mean(unique)
    rms = rmsd(unique, mean_value)
    plot_mean_with_error(path, mean_value, rms)
    if  not (dump_dir == ''):
        dump, counts = np.unique(unique, return_counts=True)
        d = dict(zip(dump, counts))
        w = csv.writer(open(dump_dir+ "hits_hist_data.csv", "w"))
        for key, val in d.items():
            w.writerow([key, val])
    # plt.axvline(mean_value, color='k', linestyle='dashed', linewidth=1)
    # min_ylim, max_ylim = plt.ylim()
    # plt.text(mean_value*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(mean_value))
    # plt.errorbar(mean_value, max_ylim*0.75, xerr = rms,capsize=3, fmt='k')
    # plt.savefig(path.replace("histogram","histogram_with_mean"))
#    print('Total number of cases: '+ str(ground_truth.shape[0]))
#    print('Hits Minimum value(%): ' + str(unique.min()))
#    print('Hits Maximum value(%): ' + str(unique.max()))

    return {
            "num" : ground_truth.shape[0],
            "min" : unique.min(),
            "max" : unique.max(),
            "mean": mean_value,
            "rms":  rms}

def plot_noise(path, predictions, ground_truth, dump_dir='', threshold = 0.5):
    '''
    Generates a histogram of noised added to the reconstructed hits

    Args:
    -----
        path: The destination to save the generated histogram
        predictions: Numpy array with the predicted hits flatenned per event
        ground_truth: Numpy array with the ground_truth hits flatenned per event
        threshold: The threshold that needs to be reached to set a value to 1
    '''

    predictions = (predictions>= threshold).astype(int)

    unique = np.zeros((ground_truth.shape[0], 1))
    i = 0
    for a1, a2 in zip(ground_truth, predictions):
        unique[i] = ((np.nonzero(a2)[0].shape[0] - np.intersect1d(np.nonzero(a1)[0], np.nonzero(a2)[0]).shape[0])/ np.nonzero(a1)[0].shape[0]) *100
        i += 1

    plt.figure()
    plt.xlabel('Noise %')
    plt.ylabel('Number of cases')
    plt.title('Noise Hits histogram')
    plt.hist(np.minimum(unique,104), bins=(np.arange(0, 110, 5)))
    plt.savefig(path)
    mean_value = mean(unique)
    rms = rmsd(unique, mean_value)
    plot_mean_with_error(path, mean_value, rms)
    if  not (dump_dir == ''):
        dump, counts = np.unique(unique, return_counts=True)
        d = dict(zip(dump, counts))
        w = csv.writer(open(dump_dir + "noise_hist_data.csv", "w"))
        for key, val in d.items():
            w.writerow([key, val])

    # plt.axvline(mean_value, color='k', linestyle='dashed', linewidth=1)
    # min_ylim, max_ylim = plt.ylim()
    # plt.text(mean_value+1, max_ylim*0.95, 'Mean: {:.2f}'.format(mean_value))
    # plt.errorbar(mean_value, max_ylim*0.75, xerr = rms,capsize=3, fmt='k')
    # plt.savefig(path.replace("histogram","histogram_with_mean"))


    return {
            "num" : ground_truth.shape[0],
            "min" : unique.min(),
            "max" : unique.max(),
            "mean": mean_value,
            "rms":  rms}

def plot_predicted_events(results_dir, noisy, prediction, i, threshold = 0.5, seed = 22):
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
    img_shape = (36,112)

    # for i, index in enumerate(img_indexes):
    plt.imsave(results_dir+'noisy_'+str(i)+'.png', noisy.reshape(img_shape))
    plt.imsave(results_dir+'denoised_'+str(i)+'.png', (prediction.reshape(img_shape)>threshold).astype(int))

def plot_noise_reduction(path, predictions, raw_input, ground_truth, dump_dir='', threshold = 0.5):
    '''
    Generates a histogram of noised added to the reconstructed hits

    Args:
    -----
        path: The destination to save the generated histogram
        predictions: Numpy array with the predicted hits flatenned per event
        raw_input: Numpy array with the raw input per event
        threshold: The threshold that needs to be reached to set a value to 1
    '''

    predictions = (predictions>= threshold).astype(int)

    unique = np.zeros((raw_input.shape[0], 1))
    init_noise_all = np.zeros((raw_input.shape[0], 1))
    rec_noise_all = np.zeros((raw_input.shape[0], 1))
    i = 0
    w = None
    if  not (dump_dir == ''):
        w = csv.writer(open(dump_dir+ "all_hits_data.csv", "w"))
        w.writerow(["sample", "true_sample", "rec", "recTrue", "recNotTrue", "noise", "hits_eff", "recNotTrue/noise", "init_noise", "rec_noise"])
    
    for a1, a2, a3 in zip(raw_input, predictions, ground_truth):
        #if((np.nonzero(a1)[0].shape[0] - np.intersect1d(np.nonzero(a1)[0], np.nonzero(a3)[0]).shape[0]) == 0):
        #    unique[i] = 100
        #else:
        if  not (dump_dir == ''):
            sample = np.nonzero(a1)[0].shape[0]
            truth_sample = np.nonzero(a3)[0].shape[0]
            rec = np.nonzero(a2)[0].shape[0]
            recTrue = np.intersect1d(np.nonzero(a2)[0], np.nonzero(a3)[0]).shape[0]
            recNotTrue = np.nonzero(a2)[0].shape[0] - np.intersect1d(np.nonzero(a2)[0], np.nonzero(a3)[0]).shape[0]
            noise = sample - truth_sample
            if truth_sample == 0:
                hit_eff = 1
            else:    
                hit_eff = recTrue/truth_sample
            if noise == 0:
                backClean = 0
            else:
                backClean = recNotTrue/noise
            init_noise = noise/truth_sample
            rec_noise = recNotTrue/truth_sample
            w.writerow([sample, truth_sample, rec, recTrue,  recNotTrue, noise, hit_eff, backClean, init_noise, rec_noise])
            init_noise_all[i] = init_noise
            rec_noise_all[i] = rec_noise
        unique[i] = ((np.nonzero(a1)[0].shape[0] - (np.nonzero(a2)[0].shape[0] ))/ (np.nonzero(a1)[0].shape[0])) *100
        # if np.nonzero(a2)[0].shape[0] > np.nonzero(a1)[0].shape[0]:
        #     plot_predicted_events("./test/", a1, a2, i)
        i += 1

    plt.figure()
    plt.xlabel('Noise Reduction %')
    plt.ylabel('Number of cases')
    plt.title('Noise Reduction histogram')
    plt.hist(unique, bins=(np.arange(0, 105, 5)))
    plt.savefig(path)
    mean_value = mean(unique)
    init_noise_mean = mean(init_noise_all)
    rec_noise_mean = mean(rec_noise_all)
    init_noise_rms = rmsd(init_noise_all, init_noise_mean)
    rec_noise_rms = rmsd(rec_noise_all, rec_noise_mean)
    rms = rmsd(unique, mean_value)
    plot_mean_with_error(path, mean_value, rms)
    if  not (dump_dir == ''):
        dump, counts = np.unique(unique, return_counts=True)
        d = dict(zip(dump, counts))
        w = csv.writer(open(dump_dir+ "noise_red_hist_data.csv", "w"))
        for key, val in d.items():
            w.writerow([key, val])

    # plt.axvline(mean_value, color='k', linestyle='dashed', linewidth=1)
    # min_ylim, max_ylim = plt.ylim()
    # plt.text(mean_value*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(mean_value))
    # plt.errorbar(mean_value, max_ylim*0.75, xerr = rms,capsize=3, fmt='k')
    # plt.savefig(path.replace("histogram","histogram_with_mean"))


    return {
            "num" : raw_input.shape[0],
            "min" : unique.min(),
            "max" : unique.max(),
            "mean": mean_value,
            "init_noise_mean": init_noise_mean,
            "init_noise_rms": init_noise_rms,
            "rec_noise_mean": rec_noise_mean,
            "rec_noise_rms": rec_noise_rms,
            "rms":  rms}
#    print('Total number of cases: '+ str(ground_truth.shape[0]))
#    print('Noise Minimum value(%): ' + str(unique.min()))
#    print('Noise Maximum value(%): ' + str(unique.max()))

def plot_hits_per_segment(path, predictions, ground_truth, num_tracks_per_event,threshold = 0.5):
    '''
    Generates 6 histograms of correctly reconstructed hits

    Args:
    -----
        path: The destination to save the generated histograms
        predictions: Numpy array with the predicted hits in a (cases, 4032) format
        ground_truth: Numpy array with the ground_truth hits in a (rows, 4032) format
        threshold: The threshold that needs to be reached to set a value to 1
    '''

    # if ground_truth.shape[1] != 4032:
    #     print('Error. Expected ground_truth to be 2 dimensional np array of with 4032 columns')
    #     exit()
    # if predictions.shape[1] != 4032:
    #     print('Error. Expected predictions to be 2 dimensional np array of with 4032 columns')
    #     exit()

    predictions = (predictions>= threshold).astype(int)
    unique = np.zeros((ground_truth.shape[0], 6, 1))

    i = 0
    ground_truth = ground_truth.reshape(-1, 6, 6, 112)
    predictions = predictions.reshape(-1, 6, 6, 112)
    valid_6 = 0
    valid_5 = 0
    valid_4 = 0
    total_num_tracks = 0
    for a1, a2, tracks in zip(ground_truth, predictions, num_tracks_per_event):
        # print(a1)
        # print(a2)
        total_num_tracks += tracks
        if tracks == 1:
            for j in range(0,6):
                for k in range(0,6):
                    temp = np.intersect1d(np.nonzero(a1[j][k])[0], np.nonzero(a2[j][k])[0]).shape[0]
                    if temp == 2:
                        temp = 1
                    unique[i][j] += temp
                
                if(unique[i][j] >= 2):
                    unique[i][j] = 1
                else:
                    unique[i][j] = 0
            
            num_valid = 0
            for j in range(0,6):
                if unique[i][j] == 1:
                    num_valid += 1
            if num_valid == 6:
                valid_6 += 1 
            elif num_valid == 5:
                valid_5 += 1 
            elif num_valid == 4 and np.sum(unique[i][2:]) == 4:
                valid_4 += 1 

            i += 1
        elif tracks == 2:
            for j in range(0,6):
                left = 0
                right = 0
                for k in range(0,6):
                    nzt = np.nonzero(a1[j][k])[0]
                    nzp = np.nonzero(a2[j][k])[0]
                    intersect = np.intersect1d(np.nonzero(a1[j][k])[0], np.nonzero(a2[j][k])[0])
                    if intersect.shape[0] > 2:
                        left += 1
                        right += 1
                    elif intersect.shape[0] == 2:
                        if(intersect[1] - intersect[0]) > 1:
                            left += 1
                            right += 1
                        elif(intersect[1] - intersect[0]) == 1:
                            if nzt.shape[0] == 2 or nzt.shape[0] == 3:
                                left += 1
                                right += 1
                            elif nzt.shape[0] == 4:
                                if intersect[0] == nzt[0]:
                                    left +=1
                                elif intersect[0] in  nzt[:2]:
                                    right +=1
                                elif intersect[0] == nzt[1]:
                                    left +=1
                                    right +=1
                    elif intersect.shape[0] == 1:
                        if nzt.shape[0] == 1:
                            left += 1
                            right += 1
                        elif nzt.shape[0] == 2:
                            if (nzt[1]  - nzt[0]) == 1:
                                left += 1
                                right += 1
                            elif nzt[0] == intersect[0]:
                                left +=1
                            elif nzt[1] == intersect[0]:
                                right +=1
                        elif nzt.shape[0] == 3:
                            if nzt[0] == intersect[0]:
                                left +=1
                            elif nzt[2] == intersect[0]:
                                right +=1
                            elif nzt[1] == intersect[0]:
                                if nzt[1] - nzt[0] == 1:
                                    left +=1
                                elif nzt[2] - nzt[1] == 1:
                                    right +=1
                        elif nzt.shape[0] == 4:
                            if nzt[0] == intersect[0] or nzt[1] == intersect[0]:
                                left +=1
                            elif nzt[2] == intersect[0] or nzt[3] == intersect[0]:
                                right +=1

                if(left>=2):
                   unique[i][j] += 1 
                if(right>=2):
                   unique[i][j] += 1 

            num_valid = 0
            for j in range(0,6):
                num_valid += unique[i][j]

            if num_valid == 12:
                valid_6 += 2 
            elif num_valid == 11:
                valid_6 += 1 
                valid_5 += 1 
            elif num_valid == 10:
                valid_6 += 1 
            elif num_valid == 9:
                valid_5 += 1 
            # elif num_valid == 4 and np.sum(unique[i][2:]) == 4:
            #     valid_4 += 1 

            i += 1

    # for j in range(0,6):
    #     plt.figure()
    #     plt.xlabel('Hit?')
    #     plt.ylabel('Number of cases')
    #     plt.title('Valid Hits histogram, Segment: '+str(j))
    #     plt.xticks(np.arange(0,2,1))
    #     plt.hist(unique[:,j], bins=2)
    #     plt.savefig(path+"."+str(j))
    #     print('Total number of cases: '+ str(ground_truth.shape[0]))
    #     print('Minimum Hits value ' + str(unique.min()))
    #     print('Maximum Hits value ' + str(unique.max()))

    return {
        "num" : total_num_tracks,
        "valid-6" : valid_6,
        "valid-5" : valid_5,
        "valid-4" : valid_4
    }

def plot_noise_per_segment(path, predictions, ground_truth, threshold = 0.5):
    '''
    Generates 6 histograms of correctly reconstructed hits

    Args:
    -----
        path: The destination to save the generated histograms
        predictions: Numpy array with the predicted hits in a (cases, 4032) format
        ground_truth: Numpy array with the ground_truth hits in a (rows, 4032) format
        threshold: The threshold that needs to be reached to set a value to 1
    '''

    if ground_truth.shape[1] != 4032:
        print('Error. Expected ground_truth to be 2 dimensional np array of with 4032 columns')
        exit()
    if predictions.shape[1] != 4032:
        print('Error. Expected predictions to be 2 dimensional np array of with 4032 columns')
        exit()

    predictions = (predictions>= threshold).astype(int)
    unique = np.zeros((ground_truth.shape[0], 6, 1))

    i = 0
    ground_truth = ground_truth.reshape(-1, 6, 6*112)
    predictions = predictions.reshape(-1, 6, 6*112)
    for a1, a2 in zip(ground_truth, predictions):
        # print(a1)
        # print(a2)
        for j in range(0,6):
            unique[i][j] = ((np.nonzero(a2[j])[0].shape[0] - np.intersect1d(np.nonzero(a1[j])[0], np.nonzero(a2[j])[0]).shape[0])/ np.nonzero(a1[j])[0].shape[0]) *100
            # if unique[i][j] >= 2:
                # unique[i][j] = 1
            # else:
                # unique[i][j] = 0
        i += 1

    for j in range(0,6):
        plt.figure()
        plt.xlabel('Noise %')
        plt.ylabel('Number of cases')
        plt.title('Noise histogram, Segment: '+str(j))
        plt.hist(unique[:,j], bins=np.concatenate((np.arange(0, 105, 5),np.array([150, 200, 250]))))
        plt.savefig(str(j)+'.'+path)
        print('Total number of cases: '+ str(ground_truth.shape[0]))
        print('Minimum Noise value ' + str(unique.min()))
        print('Maximum Noise value ' + str(unique.max()))
