from model import ML_Model
from tracking_data import TrackingData
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

def build_cnn_model():
	k_model = Sequential()
	k_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(36,112,1)))
	k_model.add(LeakyReLU(alpha=0.1))
	k_model.add(MaxPooling2D((2, 2),padding='same'))
	k_model.add(Dropout(0.25))
	k_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
	k_model.add(LeakyReLU(alpha=0.1))
	k_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
	k_model.add(Dropout(0.25))
	k_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
	k_model.add(LeakyReLU(alpha=0.1))
	k_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
	k_model.add(Dropout(0.4))
	k_model.add(Flatten())
	k_model.add(Dense(128, activation='linear'))
	k_model.add(LeakyReLU(alpha=0.1))
	k_model.add(Dropout(0.3))
	k_model.add(Dense(2, activation='softmax'))
	k_model.summary()
	k_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

	return k_model

train_data = TrackingData("./train.txt",4032)
print("Train data size:"+ str(train_data.size()))
test_data = TrackingData("./test.txt",4032)
print("Test data size:"+ str(test_data.size()))

labels = [ '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1' ]
all_a1 = []
all_ac = [] 
all_ah = [] 
all_af = []
x = np.arange(len(labels))
width = 0.20

import matplotlib.pyplot as plt

# print(labels)

for split in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 ,0.8, 0.9, 1.0]:
	
	k_model = build_cnn_model()

	clf = ML_Model(k_model)
	if split != 1.0:
		X_train,X_val,y_train,y_val = train_data.train_val_split(val_size = 1-split)
		clf.fit(X_train=X_train,X_val=X_val,y_train=y_train,y_val=y_val,epochs=2)
	else:
		clf.fit(X_train=train_data.get_all_data(),y_train=train_data.get_all_labels())

	a1,ac,ah,af = clf.get_accuracy_on_segments(test_data.get_segmented_svm_data()[0])
	all_a1.append(a1)
	all_ac.append(ac)
	all_ah.append(ah)
	all_af.append(af)
	with open("./"+str(split)+"/results.txt",'w+') as f:
		f.write("| "+str(int(split*100))+"/"+str(int((1-split)*100)) +"\n")
		f.write("| "+str(a1)+"\n")
		f.write("| "+str(ac)+"\n")
		f.write("| "+str(ah)+"\n")
		f.write("| "+str(af)+"\n")
		f.write("| "+str(clf.t_train) +"\n")
		f.write("| "+str(clf.t_infer)+"\n")

	if split != 1.0:
		clf.plot_figures("./"+str(split))

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - 3*width/2, 	all_a1, width, label='A1')
# rects2 = ax.bar(x - width/2, 	all_ac, width, label='Ac')
# rects3 = ax.bar(x + width/2, 	all_ah, width, label='Ah')
# rects4 = ax.bar(x + 3*width/2, 	all_af, width, label='Af')

# ax.set_ylabel('Score')
# ax.set_xlabel('Percentage of the training set')
# ax.set_title('Accuracy scores for different size of training set')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()

# fig.tight_layout()
# fig.savefig("try.png")