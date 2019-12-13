import tensorflow as tf
from tensorflow.keras.layers import LSTM, TimeDistributed, Activation, Bidirectional, ConvLSTM2D, Attention, Dense, Flatten, MaxPool3D, MaxPool2D,BatchNormalization, Conv3D, GRU
from tensorflow.keras import Model

class Encoder(Model):
    def __init__(self, op_dim = 30):
        super(Encoder, self).__init__()
        self.rnn = Bidirectional(LSTM(20, return_sequences= True))
        self.batchnorm = BatchNormalization()
        
        
    def call(self, inputs):
        x = self.rnn(inputs)
        x = self.batchnorm(x)
        return x


class ASRModel(Model):
    def __init__(self, op_dim = 30):
        super(ASRModel, self).__init__()
        self.encoder = Encoder()
        self.rnn = Bidirectional(LSTM(20, return_sequences= True))
        self.batchnorm = BatchNormalization()
        self.time_dense = TimeDistributed(Dense(op_dim))
        self.activation = Activation('softmax')

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.rnn(x)
        x = self.batchnorm(x)
        x = self.time_dense(x)
        x = self.activation(x) 
        return x