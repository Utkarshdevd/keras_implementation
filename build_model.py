from attention import SpatialAttention
from encoder_resnet50 import ResNet50obj
from keras.models import Model
from keras.layers import LSTM, RepeatVector, Reshape, Dense, Input, Embedding, Merge, Flatten
from keras.layers import TimeDistributed as KerasTD
from timedist import TimeDistributed
from keras.layers.core import Lambda
from keras import backend as K
from keras.layers.merge import Concatenate
from SentinelLSTMCell import SentinelLSTMCell
#from SentinelLSTMLayer import RNN as RNN_new
from SentinelRNN import RNN as RNN_new
try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing.
    pass
else:
    import sys
    sys.excepthook = IPython.core.ultratb.ColorTB()

def slice_before(x):
    return x[:,:,:512]

def slice_after(x):
    return x[:,:,:512]

def expand_dims(x):
    return K.expand_dims(x, 1)

def expand_dims_output_shape(input_shape):
    return (input_shape[0], 1, input_shape[1])

def matrix_mean(A):
    return K.mean(A, axis=1, keepdims=False)

def matrix_mean_outputShape(A_shape):
    print("A_shape:{}".format(A_shape))
    return (A_shape[0],1,A_shape[2])

sentence_length = 18
vocab_size = 10000
net50 = ResNet50obj()
resnet_model = net50.GetModel()
A = resnet_model.layers[-1].output
A_g = Lambda(matrix_mean, output_shape=matrix_mean_outputShape)(A)
V_g = Dense(2048, activation='relu')(A_g)
print(K.int_shape(A_g), K.int_shape(V_g))
V_g = Reshape((2048,))(V_g)
V_G = RepeatVector(sentence_length)(V_g)
print("V_g: {}, V_G: {}".format(K.int_shape(V_g), K.int_shape(V_G)))
# TODO: v_i(V), relu
V = KerasTD(Dense(2048, activation='relu'))(A)
V = Reshape((49*2048,))(V)
print("V: {}".format(K.int_shape(V)))
V = RepeatVector(sentence_length)(V)

words = Input(shape=(sentence_length,))
W = Embedding(output_dim=2048, input_dim=vocab_size, input_length=sentence_length)(words)
print("W: {}".format(K.int_shape(W)))
X = Concatenate(axis=-1)([V_G, W])
print("X: {}".format(K.int_shape(X)))
#h = RNN(SentinelLSTMCell(512), return_sequences=True, return_state=True)(X)
h = RNN_new(SentinelLSTMCell(512), return_sequences=True, return_state=True)(X)
for _,H in enumerate(h):
    print("H_{}: {}".format(_,K.int_shape(H)))

print("h: {}, X: {}".format(K.int_shape(h[0]), K.int_shape(X)))
s = Lambda(slice_after)(h[0])
h = Lambda(slice_before)(h[0])
print (K.int_shape(h), V, K.int_shape(s))
h = KerasTD(Dense(2048))(h)
s = KerasTD(Dense(2048))(s)
#h_expanded = Lambda(expand_dims, expand_dims_output_shape)(h)
#print (h, h_expanded, V)
attn = TimeDistributed(SpatialAttention(2048))([V,h,s])
model = Model(inputs=[resnet_model.input, words], outputs=attn)
#model.summary()
del IPython