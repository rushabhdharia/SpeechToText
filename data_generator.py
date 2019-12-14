import os
import pickle
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from string import ascii_uppercase
import numpy as np

'''
Data Generator Class - 
1. It's object returns the required input (X) and output(Y) in batches
'''

class DataGenerator(Sequence):

    '''
    Init Funtion - 
    1. initializes the path from which data is to be read
    2. initializes the lists of X and Y values to be read and sent to the model
    '''
    
    def __init__(self, path, to_fit = True):
        self.path = path
        self.list_X, self.list_Y = self.getLists()
        self.to_fit = to_fit
    
    '''
    Length Funtion - 
    1. Returns the number of time the __getitem__ function can be called
    '''

    def __len__(self):
        return len(self.list_X)
    
    '''
    Get Item Funtion - 
    1. Returns One Batch of X, true strings, input length and label length befor padding, and Y 
    '''

    def __getitem__(self, index):      
        dict_X = self.get_dict_X(index)   
        dict_Y = self.get_dict_Y(index)
        
        X, Y, input_len, label_len, y_strings = self.generate_XY(dict_X, dict_Y)
            
        return [X, y_strings, input_len, label_len], Y
    
    '''
    Get Lists Funtion - 
    1. Returns List X and List Y which contain the names of the files to be read 
    ''' 

    def getLists(self):
        list_X = []
        list_Y = []
        for item in sorted(os.listdir(self.path)):
            ext = item.split(".")[-1]
            if ext == 'pkl':
                list_X.append(item)
            elif ext == 'txt':
                list_Y.append(item)
        return list_X, list_Y
    
    '''
    Get Dictionary X Function - 
    1. Opens the respective pickle file and returns the dictionary stored in it.
    '''    

    def get_dict_X(self, index):
        file_name = self.path + self.list_X[index]
        with open(file_name, 'rb') as pickle_file:
            dict_X = pickle.load(pickle_file)
        return dict_X
    
    '''
    Get Dictionary Y Function - 
    1. Opens the respective Text file
    2. Creates a dictionary where the key is the file name and value is the true sentence
    3. returns the dictionary.
    '''        
    
    def get_dict_Y(self, index):
        filename = self.path + self.list_Y[index]
        file = open(filename)
        dict_Y = {}
        for line in file:
            data = line.split()
            key = data[0]
            value = ' '.join(data[1:])
            dict_Y[key] = value
        return dict_Y

    '''
    Generate XY Function - 
    1. First for loop - Get the maximum length of X and Y stored in the dictionaries. 
    2. Second For Loop - 
        i.   Append all true strings to the Y_string List
        ii.  Append all true input and label lengths (before padding) to their respective lists
        iii. Use the calculated max lengths for padding X and Y, so that all Xs are of the same shape and all Ys are of the same shape
        iv.  Append the padded Xs and Ys to their respective lists

    3. Stack and return the lists of X, Y, input_len, label_len, Y_strings    
    '''       
    
    def generate_XY(self, dict_X, dict_Y):
        X = []
        Y = []
        Y_strings = []
        input_len = []
        label_len = []
        
        max_x = 0
        max_y = 0
        
        for key in dict_X:
            x_temp = dict_X[key]
            y_temp = dict_Y[key]
            if max_x < x_temp.shape[1]:
                max_x = x_temp.shape[1]
            if max_y < len(y_temp):
                max_y = len(y_temp)
        
        for key in dict_X:
            x_temp = dict_X[key]
            y_temp = dict_Y[key]
            Y_strings.append(y_temp)

            input_len.append(x_temp.shape[1])
            label_len.append(len(y_temp))
            
            to_pad_x = ( (0,0), (0, max_x - dict_X[key].shape[1]))
            to_pad_y = (  (0, max_y - len(dict_Y[key])))
            
            x_temp = np.pad(dict_X[key], pad_width = to_pad_x, mode='constant', constant_values=0)
            y_temp = self.generate_Y_array(dict_Y[key], max_y)
            X.append(x_temp.T)
            Y.append(y_temp)
          
        return np.stack(X), np.stack(Y), np.stack(input_len), np.stack(label_len), Y_strings

    '''
    Generate Y Array - 
    1. Use the max len to add black tokens in the end of the string
    2. Convert the characters in the string to indices so that it can be used by the model  
    '''       
    
    def generate_Y_array(self, sentence, maxlen):
        space_token = ' '
        end_token = '>'
        blank_token = '%'
        apos_token = '\''
        
        while len(sentence) != maxlen:
            sentence += blank_token
        sentence += end_token
        
        alphabet = list(ascii_uppercase) + [space_token, apos_token, blank_token, end_token] 
        char_to_index = {}
        for idx, char in enumerate(alphabet):
            char_to_index[char] = idx

        y = []
        
        for char in sentence:
            y.append(char_to_index[char])
        
        return np.array(y)