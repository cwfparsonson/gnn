import tensorflow as tf
from tensorflow.keras.layers import Layer

class Linear(Layer):
    '''Fully-connected linear layer.'''

    def __init__(self, units=32):
        super(Linear, self).__init__()
         
        self.units = units # num units in layer

    def build(self, input_shape):
        '''Creates layer weights & biases according to shape of input initialised.'''
        # define weights
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_shape[-1], self.units), dtype='float32'),
                             trainable=True,)
        
        # define biases
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(self.units,), dtype='float32'),
                             trainable=True)

    def call(self, inputs):
        '''Performs forward propagation through layer when layer is called.

        Will automatically run build() the first time it is called.
        '''

        return tf.matmul(inputs, self.w) + self.b
