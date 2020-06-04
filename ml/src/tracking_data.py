import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import train_test_split
from joblib import dump, load

class TrackingData:

    def __init__(self, file, cols=None, one_hot = True):
        self.one_hot = one_hot
        self._load_svm_file(file,cols)


    def _load_svm_file(self,file,cols=None):
        X, y = load_svmlight_file(file,cols)
        self.X = np.array(X.todense())
        self.y = np.array(y)
        X_rows,self.X_cols = self.X.shape
        if self.one_hot:
            self.y = self._one_hot_encode(self.y)

    def train_val_split(self,val_size=0.2, seed=42):
        return train_test_split(self.X,self.y,test_size = val_size, random_state = seed)

    def _one_hot_encode(self,vec):
        '''
        For use to one-hot encode the 2-possible labels

        Args:
            vec: The labels of the data in a list
        
        Returns:
            The input vector in one-hot-encoded format (i.e. [1 0] for label 0, [0 1] for label 1)
        '''

        n = len(vec)
        out = np.zeros((n, 2))
        for i in range(n):
            out[i,int(vec[i])] = 1
        return out

    def _convert_36x112_to_36(self,input):
        """
        Given the 36x112 cols per row svm format, converts it to a 36 cols per row format  

        Args:
            input: The SVM data (without the labels) in numpy array format

        Returns:
            numpy array containing the input in 36 cols per row format
        """
        found_number = -1
        rows,cols = input.shape
        wires_hit = np.zeros((rows,36))
        jend = cols//112
        for i in range(0,rows):
            for j in range(0,jend):
                m = j*112
                for k in range(0,112):
                    if input[i][m +k] == 1:
                        if (found_number ==-1):
                            found_number = k
                        else:
                            found_number += k
                            found_number = float(found_number)/2
                            break
                    else:
                        if found_number != -1:
                            break
                wires_hit[i][j] = found_number
                found_number = -1
        return wires_hit

    def _convert_36_to_36x112(self,input):
        """
        Given the 36 cols per row svm format, converts it to a 36x112 cols per row format  

        Args:
            input: The SVM data (without the labels) in numpy array format
            
        Returns:
            numpy array containing the input in 36x112 cols per row format
        """
        rows,cols = input.shape
        wires_hit = np.zeros((rows,36*112))
        for i in range(0,rows):
            for j in range(0,cols):
                if input[i][j].is_integer():
                    wires_hit[i][j*112+int(input[i][j])] = 1
                else:
                    wires_hit[i][j*112+int(math.floor(input[i][j]))] = 1
                    wires_hit[i][j*112+int(math.ceil(input[i][j]))] = 1

        return wires_hit

    def get_segmented_svm_data(self):
        """
        Segments the passed SVM data (rows, labels) as per the instructions
        given by Gagik. Complexity is O(N). The rows must be in dense format.
        
        Args:
            rows (np.array): The rows of the SVM data in a numpy array
            labels (np.labels): The labels of the SVM data in a numpy array
        
        Returns:
            Segmented rows, segmented labels as numpy arrays
        """

        segmented_labels = []
        segmented_rows = []
        
        start_index = -1
        for current_index, val in enumerate(self.y):
            if self.one_hot:
                val = np.argmax(val,0)
            if val == 1:
                if start_index == -1:
                    start_index = current_index
                else:
                    segmented_labels.append(self.y[start_index:current_index])
                    segmented_rows.append(self.X[start_index:current_index])
                    
                    start_index = current_index

        # Add last segmented sample
        total_labels = self.y.size
        segmented_labels.append(self.y[start_index:total_labels])
        segmented_rows.append(self.X[start_index:total_labels])
        return segmented_rows, segmented_labels


    def dump_svm_file(self,file):
        """
        Store the data and labels in the current format to a file
        
        Args
            file:  The file to save the data and labels in SVM format
        
        """
        if self.one_hot:
            y = np.argmax(self.y,1)
        else:
            y = self.y
        dump_svmlight_file(np.matrix(self.X),y,file)


    def convert_format(self, data_format):
        if data_format == 0:
            if self.X_cols == 36:
                self.X = self._convert_36_to_36x112(self.X)
            elif self.X_cols == 12:
                raise Exception("Requested conversion currently not supported")
        elif data_format == 1:
            if self.X_cols == 4032:
                self.X = self._convert_36x112_to_36(self.X)
            elif self.X_cols == 12:
                raise Exception("Requested conversion currently not supported")
        elif data_format == 2:
            raise Exception("Requested conversion currently not supported")
        else:
            raise Exception('data_format should be 1,2 or 3 . Value was: {}'.format(data_format))

    def get_all_labels(self):
        return self.y

    def get_all_data(self):
        return self.X

    def size(self):
        return len(self.X)

