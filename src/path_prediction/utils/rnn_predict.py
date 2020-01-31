from tracking_data import TrackingData
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
import numpy as np
import sys
print("Importing Tensorflow")
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import fileinput


path_train = "/home/aange002/code/JLab-ML/clas12-ml/data/36/training/data_10-07-2019.txt"
path_test = "/home/aange002/code/JLab-ML/clas12-ml/data/36/testing/data_10-07-2019.txt"
path_model = "./out_model.h5"

def remove_zeros(dataset):
    for v in dataset:
        for i in range(0,36):
            if v[i] == 0:
                if i > 0 and i<35:
                    v[i] = (v[i-1] + v[i+1])/2
                elif i == 0:
                    v[i] = v[i+1]
                elif i == 35:
                    v[i] = v[i-1]
    return dataset

def round_to_nearset_half(numbers):
    for i in range(0,numbers.shape[0]):
        numbers[i] =  round (numbers[i])
    return numbers


def generated_data_loss(y_gen,y_true,num_gen):
    curr_loss = 0
    N = y_gen.shape[0]*num_gen
    for yp,yt in zip (y_gen[:,-num_gen:],y_true[:,-num_gen:]):
        for y1,y2 in zip(yp,yt):
            curr_loss += math.fabs(y1-y2)
    return curr_loss/N


print("Loading Train and Test Data")
train_data = TrackingData(path_train,36,False)
test_data = TrackingData(path_test,36,False)


train_valid_rows = remove_zeros( train_data.get_valid_rows())

test_valid_rows = remove_zeros( test_data.get_valid_rows())



train_valid_rows =  np.array([x/111 for x in train_valid_rows])
test_valid_rows = np.array([x/111 for x in test_valid_rows])

y_train = train_valid_rows[:,12:36]
y_train = y_train.reshape(-1,24)

X_train = train_valid_rows[:,:24]
X_train = X_train.reshape(-1,24,1)

y_test = test_valid_rows[:,12:36]
y_test = y_test.reshape(-1,24)

X_test = test_valid_rows[:,:24]
X_test = X_test.reshape(-1,24,1)


model = tf.keras.models.load_model(path_model)
model.summary()

X_test_all = test_valid_rows[:,:24]
predictions = model.predict(X_test.reshape(-1,24,1))
X_test_all = np.column_stack((X_test_all,predictions[:,-12:]))
X_test_all = np.array([round_to_nearset_half(x*111) for x in X_test_all])
test_valid_rows = np.array([round_to_nearset_half(x*111) for x in test_valid_rows])
# print(X_test_all[0])
print("Loss from generated data only:")
print(generated_data_loss(X_test_all,test_valid_rows,12))

while (True):
    example = int(input('Example to show: '))
    if example >=0 and example<X_test.shape[0]:
        img1 = test_data._convert_36_to_36x112(remove_zeros(test_data.get_valid_rows())[example]).reshape(36,112)
        img2 = test_data._convert_36_to_36x112(X_test_all[example]).reshape(36,112)
        img2[-12:] = img2[-12:]/2
        img3 = np.array(img1)

        for k in range(0,img2.shape[0]):
            for i in range(0,img2.shape[1]):
                    if img2[k,i] != 0:
                        img3[k,i] = img2[k,i]

        f, axarr = plt.subplots(2)
        axarr[0].imshow(img1)
        axarr[1].imshow(img3)
        plt.show()
        plt.savefig("overlapped_trajectories.png")
    else:
        break
