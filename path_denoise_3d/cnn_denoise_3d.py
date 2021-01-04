import sys
import time

from sklearn.datasets import load_svmlight_file
import numpy as np
print('Importing tensorflow')
import tensorflow as tf
print('Done importing tensorflow')
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
print('Matplotlib')
import os.path
from sklearn.model_selection import train_test_split
import random
import sys
sys.path.append('../path_denoise')
import histogram

TOTAL_PLANES = 24
RINGS_PER_PLANE = 21
PADDING = 6
PADS_PER_RING = 122 + PADDING

train_data = load_svmlight_file('data/denoising/train_250.lsvm', TOTAL_PLANES * RINGS_PER_PLANE * PADS_PER_RING)
# test_data = load_svmlight_file('data/2020/11/driftchamber_multitracks_test.lsvm',4032)
# train_data = load_svmlight_file('data/2020/10/23/train.lsvm',4032)
# test_data = load_svmlight_file('data/2020/10/23/test.lsvm',4032)
# test_data = load_svmlight_file('data/2020/11/driftchamber_multitracks_2_or_more_test.lsvm',4032)
# print(np.array(train_data[0].todense()))
# print(np.array(train_data[1]))

all_valid = []
all_invalid = []

def loss_ssim(y_true, y_pred):
    # return 1
    return 2 - tf.reduce_mean(1 + tf.image.ssim(y_true, y_pred, 1.0))


for v,array in zip(train_data[1], train_data[0].todense()):
    if v == 1:
        all_valid.append(array)
    else:
        all_invalid.append(array)


print(len(all_valid))
print(len(all_invalid))

# for v,array in zip(test_data[1], test_data[0].todense()):
#     if v == 1:
#         test_valid.append(array)
#     else:
#         test_invalid.append(array)

# print(valid)
all_valid = np.vstack(all_valid)
all_invalid = np.vstack(all_invalid)
del train_data
# test_valid = np.vstack(test_valid)
all_valid = np.asarray(all_valid, dtype=np.float16)
all_invalid = np.asarray(all_invalid, dtype=np.float16)
# test_invalid = np.vstack(test_invalid)

# DEBUG CODE ############
#all_valid =  all_valid.reshape(-1, TOTAL_PLANES * RINGS_PER_PLANE, PADS_PER_RING)
#plt.imsave('test.png',all_valid[0].reshape(TOTAL_PLANES * RINGS_PER_PLANE, PADS_PER_RING))
#exit()
# DEBUG CODE ############

X_train, X_test, y_train, y_test = train_test_split(all_invalid, all_valid, test_size=0.1, random_state=42)

model = None
history = None
train_time =0
choice = 1

# test_invalid = np.array(test_invalid).reshape(-1, 36, 112, 1)
# test_valid = np.array(test_valid).reshape(-1, 36, 112, 1)
X_train = X_train.reshape(-1, TOTAL_PLANES * RINGS_PER_PLANE, PADS_PER_RING, 1)
X_test = X_test.reshape(-1, TOTAL_PLANES * RINGS_PER_PLANE, PADS_PER_RING, 1)
y_train = y_train.reshape(-1, TOTAL_PLANES * RINGS_PER_PLANE, PADS_PER_RING, 1)
y_test = y_test.reshape(-1, TOTAL_PLANES * RINGS_PER_PLANE, PADS_PER_RING, 1)
if os.path.exists('./cnn_autoenc_90_10.h5') == False:
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 4), activation='relu', padding="same", input_shape=(TOTAL_PLANES * RINGS_PER_PLANE, PADS_PER_RING, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 4), activation='relu', padding="same"))
    model.add(tf.keras.layers.MaxPooling2D((3, 2)))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 4), activation='relu', padding="same"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 4), activation='relu', padding="same"))
    model.add(tf.keras.layers.UpSampling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 4), activation='relu', padding="same"))
    model.add(tf.keras.layers.UpSampling2D((3, 2)))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 4), activation='relu', padding="same"))
    model.add(tf.keras.layers.UpSampling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(1, kernel_size=(3, 4), activation='sigmoid', padding="same"))
    model.summary()

    # if use_ssim:
    #     model.compile(optimizer='adam', loss=loss_ssim)
    # else:
    model.compile(optimizer='nadam', loss='binary_crossentropy')

    start = time.time()
    history = model.fit(x=X_train, y=y_train, batch_size=1, epochs=20, verbose=True, validation_data=(X_test, y_test))
    end = time.time()
    train_time = end - start

    model.save('cnn_autoenc_90_10.h5')
else:
    print("Loading existing model")
    model = tf.keras.models.load_model('cnn_autoenc_90_10.h5')
    print("Done")

start = time.time()
model.evaluate(X_test, y_test, verbose=True, batch_size=16)
end = time.time()

print("Time to infer:" + str((end-start)/X_test.shape[0]))
print("Time to train:" + str(train_time))
arr = (model.predict(X_test))
sample = (arr[:]> 0.5) .astype(int)
for i in range(0,6):
    rand = random.randint(0, y_test.shape[0])
    plt.imsave('correct'+str(i)+'.png',y_test[rand].reshape(TOTAL_PLANES * RINGS_PER_PLANE, PADS_PER_RING))
    plt.imsave('noisy'+str(i)+'.png',X_test[rand].reshape(TOTAL_PLANES * RINGS_PER_PLANE, PADS_PER_RING))
    plt.imsave('cnn_denoised'+str(i)+'.png',sample[rand].reshape(TOTAL_PLANES * RINGS_PER_PLANE, PADS_PER_RING))
sample = arr[1:2]

if history:
    training_loss = history.history['loss']
    test_loss = history.history['val_loss']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)
    plt.figure(1)

    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, test_loss, 'b--')
    plt.yticks(np.arange(0, training_loss[0], step=0.005))
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('cnn_train_val_loss.png')
    print("loss: " + str(training_loss[-1]))
    print("val_loss: " + str(test_loss[-1]))


truth = y_test.reshape(-1, TOTAL_PLANES * RINGS_PER_PLANE * PADS_PER_RING)
derived = (arr.reshape(-1, TOTAL_PLANES * RINGS_PER_PLANE * PADS_PER_RING))
histogram.plot_hits('hits_histogram.png', derived, truth)
histogram.plot_noise('noise_histogram.png', derived, truth)
