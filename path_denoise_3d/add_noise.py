import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import dump_svmlight_file

PADDING = 6

def sp_noise(images,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(images.shape,np.uint8)
    thres = 1 - prob
    for k in range(images.shape[0]):
        output[k] = images[k]
        for i in range(images[k].shape[0]):
            max_noise_pts = random.randint(0, 6)
            rdn = random.random()
            if rdn > thres:
                j = random.randint(0,121)
                for _ in range(0, max_noise_pts):
                    output[k][random.randint(max(0, i-3), min(images[k].shape[0]-1, i+3))][random.randint(max(0, j-3),min(j+3,121))] = 1
    return output


print('Loading numpy array')
array = np.load('np_250.npy')
print(array.shape)
print('Loaded numpy array')

imgs = array.reshape(-1, array.shape[1] * 21, 122)
plt.imsave('single_250track.png',imgs[0])
noisy = sp_noise(imgs, 0.45)
# img = np.logical_or( img, noise)
plt.imsave('single_track_noisy_250.png',noisy[0])
train = np.concatenate((imgs, noisy))
labels = np.concatenate((np.ones((imgs.shape[0], )), np.zeros((noisy.shape[0],))))

train = np.pad(train, [(0, 0), (0, 0), (0, PADDING)], mode='constant')
dump_svmlight_file(train.reshape(-1, array.shape[1]*21*(122 + PADDING)), labels, 'diff_noise_train_250.lsvm')
