from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class SpatialAttention(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(SpatialAttention, **kwargs)

    def build(self, input_shape):
        k = input_shape[1]-1
        d = input_shape[-1]
        print "k: {}".format(k)
        self.w_h = K.random_uniform_variable(shape=(k))
        self.W_v = K.random_uniform_variable(shape=(k, d))
        self.W_g = K.random_uniform_variable(shape=(k, d))
        self.ones = K.ones(k)

        super(SpatialAttention, self).build(input_shape)
    
    def call(self, x):
        v = x[:-1]
        h = x[-1]
        z_t = self.w_h * K.tanh(self.W_v * v + self.W_g * h_t * K.transpose(self.ones))
        alpha_t = K.softmax(z_t)
        return alpha_t