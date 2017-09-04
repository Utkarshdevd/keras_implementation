from keras.applications.vgg16 import VGG16
from keras.models import Model

vgg_model = VGG16(include_top=True, weights='imagenet')

# Disassemble layers
layers = [l for l in vgg_model.layers]

x = layers[0].input
for i in range(1, len(layers)-2):
    layers[i].trainable = False
    x = layers[i](x)

# result model has all layer, expect the last fc and softmax/prediction layers(as proposed in SalGAN)
result_model = Model(input=layers[0].input, output=x)
result_model.summary()

return result_model