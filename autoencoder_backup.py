class DataGenerator_AutoEncoder(Sequence):
    def __init__(self, path, to_fit = True):
        self.path = path
        self.list_X = self.getList()
        self.to_fit = to_fit
    
    def __len__(self):
        return len(self.list_X)
    
    def __getitem__(self, index):      
        dict_X = self.get_dict_X(index)
        X = np.stack(self.generate_X(dict_X), axis = 0)
        if self.to_fit:    
            return X, X
        
        return X
    
    def getList(self):
        train_list = os.listdir(self.path)
        list_X = [item for item in train_list if item.split(".")[-1] == 'pkl']
        return list_X
    
    def get_dict_X(self, index):
        file_name = self.path + self.list_X[index]
        with open(file_name, 'rb') as pickle_file:
            dict_X = pickle.load(pickle_file)
        return dict_X
    
    def generate_X(self, dict_X):
        X = [value.T for key, value in dict_X.items()]
        X = tf.keras.preprocessing.sequence.pad_sequences(X, padding = 'post')
        
        return X


# Need to flatten
def autoencoder():
    model = tf.keras.Sequential()
    model.add(layers.Dense(350))
    model.add(layers.Dropout(rate = 0.2))
    model.add(layers.Dense(200))
    model.add(layers.Dropout(rate = 0.2))
    model.add(layers.Dense(100))
    model.add(layers.Dropout(rate = 0.2))
    model.add(layers.Dense(200))
    model.add(layers.Dropout(rate = 0.2))
    model.add(layers.Dense(350))
    model.add(layers.Dropout(rate = 0.2))
    model.add(layers.Dense(513))    
    return model

model = autoencoder()
model.compile(optimizer = tf.keras.optimizers.Adam(), loss = tf.keras.losses.MeanSquaredError())
training_generator = DataGenerator_AutoEncoder(train_path)
validation_generator = DataGenerator_AutoEncoder(dev_path)
model.fit_generator(generator=training_generator, validation_data=validation_generator,epochs=20)