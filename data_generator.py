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
	
	def __init__(self, path, batch_size = 4,to_fit = True):
		self.path = path
		self.list_X, self.list_Y = self.getLists()
		self.all_dict_X = self.get_all_X()
		self.all_dict_Y = self.get_all_Y() 
		self.key_list = [*self.all_dict_X]
		# assert len(all_dict_X) == len(all_dict_Y), "dictionary lengths do not match"
		self.to_fit = to_fit
		self.batch_size = batch_size

	'''
	Length Funtion - 
	1. Returns the number of time the __getitem__ function can be called
	'''

	def __len__(self):
		return len(self.key_list)//self.batch_size
	
	'''
	Get Item Funtion - 
	1. Returns One Batch of X, true strings, input length and label length befor padding, and Y 
	'''

	def __getitem__(self, index):      
		start = self.batch_size*index
		end = start + self.batch_size

		keys = self.key_list[start:end]
		X, Y, input_len, label_len, y_strings = self.generate_XY(keys)
			
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

	def get_all_X(self):
		super_dict = {}
		for index in range(len(self.list_X)):
			d = self.get_dict_X(index) 
			for key, value in d.items():
				super_dict[key] = value

		return super_dict

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

	def get_all_Y(self):
		super_dict = {}
		for index in range(len(self.list_Y)):
			filename = self.path + self.list_Y[index]
			file = open(filename)
			for line in file:
				data = line.split()
				key = data[0]
				value = ' '.join(data[1:])
				super_dict[key] = value
				
		return super_dict      


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
	
	def generate_XY(self, keys):
		X = []
		Y = []
		Y_strings = []
		input_len = []
		label_len = []
		
		max_x = 0
		max_y = 0
		
		for key in keys:
			shape_x = self.all_dict_X[key].shape[1]
			if max_x < shape_x:
				max_x = shape_x

			shape_y = len(self.all_dict_Y[key])
			if max_y< shape_y:
				max_y = shape_y

		for key in keys:
			x_temp = self.all_dict_X[key]
			y_temp = self.all_dict_Y[key]
			Y_strings.append(y_temp)

			shape_x = x_temp.shape[1]
			shape_y = len(y_temp)

			input_len.append(shape_x)
			label_len.append(shape_y)    

			to_pad_x = ( (0,0), (0, max_x - shape_x))
			to_pad_y = (  (0, max_y - shape_y))      

			x_temp = np.pad(x_temp, pad_width = to_pad_x, mode='constant', constant_values=0)
			y_temp = self.generate_Y_array(y_temp, max_y)
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