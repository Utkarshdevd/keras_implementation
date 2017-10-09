from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Reshape
import numpy as np

class SpatialAttention(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(SpatialAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        print(input_shape)
        k = 49
        d = input_shape[-1][1]
        print("k: {}\nd: {}".format(k,d))

        self.w_h = self.add_weight(name='w_h', shape=(k, 1,), initializer='uniform', trainable=True)
        self.W_v = self.add_weight(name='W_v',shape=(k, d), initializer='uniform', trainable=True)
        self.W_g = self.add_weight(name='W_g',shape=(k, d), initializer='uniform', trainable=True)
        self.ones = K.ones(shape=(k,1))

        self.w_h_s = self.add_weight(name='w_h', shape=(1, k,), initializer='uniform', trainable=True)
        self.W_s = self.add_weight(name='W_g',shape=(k, d), initializer='uniform', trainable=True)
        self.k = k

        super(SpatialAttention, self).build(input_shape)
    
    def call(self, inputs, **kwargs):
        V = Reshape((49,2048))(inputs[0])
        h_t = inputs[1]
        s_t = inputs[2]
        print(("V:{}\nh:{}\ns_t:{}").format(K.int_shape(V), K.int_shape(h_t), K.int_shape(s_t)))
        W_vV = K.dot(V, K.transpose(self.W_v))
        print ("W_vV: {}".format(K.int_shape(W_vV)))
        W_gt = K.dot(h_t, K.transpose(self.W_g))
        print ("W_gt: {}".format(K.int_shape(W_gt)))
        z_t = K.dot(K.tanh(K.dot(W_gt, self.ones) + W_vV), self.w_h)
        #z_t = K.
        print ("z: {}".format(K.int_shape(z_t)))
        W_st = K.tanh(K.dot(s_t, K.transpose(self.W_s))+W_gt)
        print ("W_st: {}".format(K.int_shape(W_st)))
        sft_beta_t = K.dot(self.w_h_s, K.transpose(W_st))
        print ("sft: {}".format(K.int_shape(sft_beta_t)))
        alpha_cap_t = K.softmax([z_t, sft_beta_t])
        alpha_cap_t = K.squeeze(alpha_cap_t[-1:], axis=2)
        alpha_t = alpha_cap_t[:-1]
        beta_t = alpha_cap_t[-1]
        print ("alpha: {}".format(alpha_t))
        c_t = K.batch_dot(alpha_t, V, axes=1)
        print ("c_t: {}".format(c_t))
        c_cap_t = beta_t * s_t + (1-beta_t) * c_t
        return c_cap_t + h_t

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
