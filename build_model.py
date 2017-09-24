from attention import SpatialAttention
from encoder_resnet50 import ResNet50obj
from keras.models import Model
from keras.layers import LSTM, concatenate, RepeatVector, Reshape, Dense
from keras.layers import TimeDistributed as KerasTD
from timedist import TimeDistributed
from keras.layers.core import Lambda
from keras import backend as K

try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing.
    pass
else:
    import sys
    sys.excepthook = IPython.core.ultratb.ColorTB()

def expand_dims(x):
    return K.expand_dims(x, 1)

def expand_dims_output_shape(input_shape):
    return (input_shape[0], 1, input_shape[1])

net50 = ResNet50obj()
resnet_model = net50.GetModel()
A = resnet_model.layers[-1].output
# TODO: a_g, use mean
A_g = K.mean(A, axis=1)
# TODO: v_g, relu
V_g = Dense(2048, activation='relu')(A_g)
print(K.int_shape(A_g), K.int_shape(V_g))
# TODO: v_i(V), relu
V = KerasTD(Dense(2048, activation='relu'))(A)
V = Reshape((49*2048,))(V)
V = RepeatVector(18)(V)
print(K.int_shape(V))
h = LSTM(2048, return_sequences=True)(V)
print(K.int_shape(h))
print (h, V)
#h_expanded = Lambda(expand_dims, expand_dims_output_shape)(h)
#print (h, h_expanded, V)
attn = TimeDistributed(SpatialAttention(2048))([V,h])
model = Model(inputs = resnet_model.input, outputs = attn)
#model.summary()