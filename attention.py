from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class SpatialAttention(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(SpatialAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        print(input_shape)
        k = input_shape[0][1]
        d = input_shape[-1][1]
        print("k: {}\nd: {}".format(k,d))

        self.w_h = self.add_weight(name='w_h', shape=(k, 1,), initializer='uniform', trainable=True)
        self.W_v = self.add_weight(name='W_v',shape=(k, d), initializer='uniform', trainable=True)
        self.W_g = self.add_weight(name='W_g',shape=(k, d), initializer='uniform', trainable=True)
        self.ones = K.ones(shape=(k,1))
        self.k = k

        super(SpatialAttention, self).build(input_shape)
    
    def call(self, inputs, **kwargs):
        V = inputs[0]
        h_t = inputs[1]
        #print(("V:{}\nh:{}\n{}").format(h_t, self.W_v))
        W_vV = K.dot(V, K.transpose(self.W_v))
        print ("W_vV: {}".format(W_vV))
        W_gt = K.dot(h_t, K.transpose(self.W_g))
        print ("W_gt: {}".format(W_gt))
        z_t = K.dot(K.tanh(K.dot(W_gt, self.ones) + W_vV), self.w_h)
        print ("z: {}".format(z_t))
        alpha_t = K.softmax(z_t)
        alpha_t = K.squeeze(alpha_t, axis=2)
        print ("alpha: {}".format(alpha_t))
        c_t = K.batch_dot(alpha_t, V, axes=1)
        print ("c_t: {}".format(c_t))
        return alpha_t

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
