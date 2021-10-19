from termcolor import colored
from models.AbstractKerasClassifier import AbstractKerasClassifier
import numpy as np
import tensorflow 
# import tensorflow.keras
from tensorflow.keras import Sequential,Input,Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU

class CnnModel(AbstractKerasClassifier):

    def __init__(self, in_dict):
        super().__init__(in_dict)
        self.preprocess_input(in_dict)
        self.model = None

#    def __init__(self):
#        super().__init__()
#        self.model = None

    def build_new_model(self):
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
        k_model.add(Dense(self.n_classes, activation='softmax'))
        k_model.summary()
        k_model.compile(loss=tensorflow.keras.losses.categorical_crossentropy, optimizer=tensorflow.keras.optimizers.Adam(),metrics=['accuracy'])
        self.model = k_model

    def one_hot_encode(self,vec):
        '''
        For use to one-hot encode the 2-possible labels

        Args:
            vec: The labels of the data in a list
        
        Returns:
            The input vector in one-hot-encoded format (i.e. [1 0] for label 0, [0 1] for label 1)
        '''

        n = len(vec)
        print(self.n_classes)
        out = np.zeros((n, self.n_classes))
        for i in range(n):
            out[i,int(vec[i])] = 1
        return out

    def preprocess_input(self,input_dict):
        super().preprocess_input(input_dict)

        if "training" in input_dict:
            X_train = np.array(input_dict["training"]["data"]).reshape(-1,36,112,1)
            input_dict["training"]["data"] = X_train
            input_dict["training"]["labels"] = self.one_hot_encode(input_dict["training"]["labels"])
        X_test = np.array(input_dict["testing"]["data"]).reshape(-1,36,112,1)
        input_dict["testing"]["data"] = X_test 
        input_dict["testing"]["labels"] = self.one_hot_encode(input_dict["testing"]["labels"])


    def train(self, input_dict) -> dict:
        print(colored("Training CNN model...", "green"))
        return super().train(input_dict)

    def test(self, input_dict) -> dict:
        print(colored("Testing CNN model...", "green"))
        return super().test(input_dict)

    def predict(self, input_dict) -> dict:
        data = input_dict['prediction']['data'] 
        if (data.shape[-1] != 112 or data.shape[-2] != 36):
            data = data.reshape(-1,36,112,1)
        input_dict['prediction']['data'] = data
        
        return super().predict(input_dict)
