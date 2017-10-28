from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Reshape, Flatten
from keras import backend as K
from keras.optimizers import SGD

class ResNet50obj(object):
    def __init__(self):            
        resnet_model = ResNet50(weights='imagenet')

        resnet_model.layers.pop()
        resnet_model.layers.pop()
        resnet_model.layers.pop()

        #resnet_model = Reshape((49, 2048)) (resnet_model.output)
        x = resnet_model.layers[-1].output
        reshaped = Reshape((49,2048))(x)
        model = Model(inputs=resnet_model.input, outputs=reshaped)

        for _,layer in enumerate(model.layers[:172]):
            layer.trainable = False
            #print ("{} layerName : {} layerDim: {} {}".format(_, layer.name, layer.output_shape, layer.trainable))
        for _,layer in enumerate(model.layers[172:]):
            layer.trainable = True
            #print ("{} layerName : {} layerDim: {} {}".format(_, layer.name, layer.output_shape, layer.trainable))
        
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
        #model.summary()

        self.result_model = model

    def GetModel(self):
        return self.result_model

def main():
    resnet = ResNet50obj()        

if __name__ == '__main__':
    main()