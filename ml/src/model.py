import json
import time
from sklearn.base import ClassifierMixin
import numpy as np
import matplotlib.pyplot as plt

class ML_Model():
    def __init__(self,model=None):
        self.model = None
        self.keras_model = False
        
        if model != None:
            self.model = model
            if type(model) == ClassifierMixin: 
                print("Imported sklearn Model")
                self.keras_model = False
            else:
                print("Imported Keras Model")
                self.keras_model = True
                import keras



    def save(self,model_file=None,history_file=None):
        if self.keras_model:
            self.model.save(model_file)
            if self.history:
                json.dump(self.history, open(history_file,'w'))
        else:
            dump(self.model,model_file)

    def fit(self,X_train,y_train,X_val=None,y_val=None,epochs=20,batch_size=32):
        if self.model == None:
            raise Exception('Error fitting model. No model has been chosen ')

        t1 = time.time()
        if self.keras_model == True:
            if not (X_val is None):
                self.history = self.model.fit(X_train.reshape(-1,36,112,1), y_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_val.reshape(-1,36,112,1), y_val)).history
                self.val_acc = self.history['val_acc']
            else:
                self.history = self.model.fit(X_train.reshape(-1,36,112,1), y_train, batch_size=batch_size,epochs=epochs,verbose=1).history
            t2 = time.time() - t1

        else:
            if not (X_val is None):
                self.model.fit(X_train, y_train)
                self.val_acc = self.model.score(X_val,y_val)
            else:
                self.model.fit(X_train, y_train)

            t2 = time.time() - t1
        self.t_train = t2

    def plot_figures(self,directory):
        if self.keras_model:
            accuracy = self.history['acc']
            val_accuracy = self.history['val_acc']
            loss = self.history['loss']
            val_loss = self.history['val_loss']
            epochs = range(len(accuracy))
            fig1 = plt.figure()
            plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
            plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
            plt.title('Training and validation accuracy')
            plt.legend()
            fig2 = plt.figure()
            plt.plot(epochs, loss, 'bo', label='Training loss')
            plt.plot(epochs, val_loss, 'b', label='Validation loss')
            plt.title('Training and validation loss')
            plt.legend()
            fig1.savefig(directory+'/cnn_acc.png')
            fig2.savefig(directory+'/cnn_loss.png')


    def validation(self):
        return self.val_acc


    def get_accuracy_on_segments(self,segmented_data):
        t2 = 0.0
        correct = 0
        fp = 0
        hp = 0
        fail = 0
        all_predictions = []
        for sample in segmented_data:
            if self.keras_model:
                temp = sample.reshape(-1,36,112,1)
                t1 = time.time()
                prediction_prob = self.model.predict(temp)
                t2 = t2 + time.time() - t1
                prediction = np.argmax(prediction_prob,1)
            else:
                t1 = time.time()
                prediction_prob = self.model.predict_proba(sample)
                t2 = t2 + time.time() - t1
                prediction = np.argmax(prediction_prob,1)
            if prediction[0] == 1:
                correct +=1
                for val in prediction[1:]:
                    if val == 1:
                        fp +=1
                if np.argmax(prediction_prob[:,1],0) == 0:
                    hp +=1
            else:
                fail += 1

        self.a1 = correct/len(segmented_data)
        self.ac = fp/len(segmented_data)
        self.ah = hp/len(segmented_data)
        self.af = fail/len(segmented_data)
        self.t_infer = t2/len(segmented_data)

        return self.a1,self.ac,self.ah,self.af

    def load_sklearn(self,model_file,history_file=None):
        self.model = load(model_file)
    
    def load_keras(self,model_file,history_file=None):
        import keras
        from keras.models import load_model

        self.model = load_model(model_file)
        self.model.summary()

        if history_file:
            self.history = json.load(open(history_file,'r'))
