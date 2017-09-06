from attention import SpatialAttention
from encoder_resnet50 import ResNet50obj
from keras.models import Model
from keras.layers import LSTM, concatenate
from keras.layers.core import Lambda
from keras import backend as K

def expand_dims(x):
    return K.expand_dims(x, 1)

def expand_dims_output_shape(input_shape):
    return (input_shape[0], 1, input_shape[1])

net50 = ResNet50obj()
resnet_model = net50.GetModel()
V = resnet_model.layers[-1].output
h = LSTM(2048)(V)
print (h, V)
#h_expanded = Lambda(expand_dims, expand_dims_output_shape)(h)
#print (h, h_expanded, V)
attn = SpatialAttention(2048)([V,h])
model = Model(inputs = resnet_model.input, outputs = attn)
#model.summary()