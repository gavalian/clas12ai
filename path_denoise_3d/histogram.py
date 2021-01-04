import numpy as np
import matplotlib.pyplot as plt


def plot_hits(path, predictions, ground_truth, threshold = 0.5):
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
#    print('Total number of cases: '+ str(ground_truth.shape[0]))
#    print('Hits Minimum value(%): ' + str(unique.min()))
#    print('Hits Maximum value(%): ' + str(unique.max()))
    
    return {  
            "num" : ground_truth.shape[0], 
            "min" : unique.min(), 
            "max" : unique.max()}

def plot_noise(path, predictions, ground_truth, threshold = 0.5):
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
    plt.hist(unique, bins=np.concatenate((np.arange(0, 105, 5),np.array([150, 200, 250]))))
    plt.savefig(path)
    
    return {
            "num" : ground_truth.shape[0],
            "min" : unique.min(),
            "max" : unique.max()}
#    print('Total number of cases: '+ str(ground_truth.shape[0]))
#    print('Noise Minimum value(%): ' + str(unique.min()))
#    print('Noise Maximum value(%): ' + str(unique.max()))

def plot_hits_per_segment(path, predictions, ground_truth, threshold = 0.5):
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
            unique[i][j] = np.intersect1d(np.nonzero(a1[j])[0], np.nonzero(a2[j])[0]).shape[0]
            if unique[i][j] >= 2:
                unique[i][j] = 1
            else:
                unique[i][j] = 0
        i += 1

    for j in range(0,6):
        plt.figure()
        plt.xlabel('Hit?')
        plt.ylabel('Number of cases')
        plt.title('Valid Hits histogram, Segment: '+str(j))
        plt.xticks(np.arange(0,2,1))
        plt.hist(unique[:,j], bins=2)
        plt.savefig(str(j)+'.'+path)
        print('Total number of cases: '+ str(ground_truth.shape[0]))
        print('Minimum Hits value ' + str(unique.min()))
        print('Maximum Hits value ' + str(unique.max()))

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
