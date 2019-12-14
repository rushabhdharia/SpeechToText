import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPool2D, LSTM, Flatten, Bidirectional, LayerNormalization, Dense, TimeDistributed, Activation
from tensorflow.keras import Model

class Encoder(Model):
    def __init__(self, op_dim = 30):
        super(Encoder, self).__init__()
        self.cnn_1 = Conv2D(4, kernel_size = (1, 7), padding = 'same')
        self.maxpool = MaxPool2D(pool_size = (1, 2))
        self.cnn_2 = Conv2D(8, kernel_size = (1, 5), padding = 'same')
        self.cnn_3 = Conv2D(16, kernel_size = (1, 3), padding = 'same')
        self.layernorm_1 = LayerNormalization()
        self.layernorm_2 = LayerNormalization()
        self.layernorm_3 = LayerNormalization()
                

    def call(self, inputs):
        x = tf.expand_dims(inputs, axis = 3)
        x = self.cnn_1(x)
        x = self.maxpool(x)
        x = self.layernorm_1(x)
        x = self.cnn_2(x)
        x = self.maxpool(x)
        x = self.layernorm_2(x)
        x = self.cnn_3(x)
        x = self.maxpool(x)
        x = self.layernorm_3(x)
        return x


class ASRModel(Model):
    def __init__(self, op_dim = 30):
        super(ASRModel, self).__init__()
        self.encoder = Encoder()
        self.rnn = LSTM(16, return_sequences = True) #64 cloud
        self.layernorm = LayerNormalization()
        self.time_dense = TimeDistributed(Dense(op_dim))
        self.activation = Activation('softmax')


    def call(self, inputs):
        x = self.encoder(inputs)
        batchsize, time_seq, width, channels = x.shape
        x = tf.reshape(x, shape = [batchsize, time_seq, width * channels])
        x = self.rnn(x)
        x = self.layernorm(x)
        x = self.time_dense(x)
        x = self.activation(x) 
        return x