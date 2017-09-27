# -*- coding: utf-8 -*-

from keras import backend as K
from keras.layers import LSTM

class MyLSTM(LSTM):
    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._generate_dropout_mask(inputs, training=training)
        self.cell._generate_recurrent_dropout_mask(inputs, training=training)
        self.extra_output = states
        return super(LSTM, self).call(inputs,
                                      mask=mask,
                                      training=training,
                                      initial_state=initial_state)
I = Input(shape=(...))
lstm = MyLSTM(20)
output = lstm(I) # by calling, we actually call the `call()` and create `lstm.extra_output`
extra_output = lstm.extra_output # refer to the target

calculate_function = K.function(inputs=[I], outputs=extra_output+[output]) # use function to calculate them **simultaneously**