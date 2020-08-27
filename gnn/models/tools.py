import dgl
import tensorflow as tf
from tensorflow.keras.layers import Layer
import sys

class Linear(Layer):
    '''Fully-connected linear layer.'''

    def __init__(self, 
                 units=32,
                 bias=True):
        super(Linear, self).__init__()
         
        self.units = units # num units in layer
        self.bias = bias # whether or not to use bias units in layer

    def build(self, input_shape):
        '''Creates layer weights & biases according to shape of input initialised.'''
        # define weights
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_shape[-1], self.units), dtype='float32'),
                             trainable=True,)
        
        # define biases
        if self.bias:
            b_init = tf.zeros_initializer()
            self.b = tf.Variable(initial_value=b_init(shape=(self.units,), dtype='float32'),
                                 trainable=True)
        else:
            self.b = None

    def call(self, inputs, activation=None):
        '''Performs forward propagation through layer when layer is called.

        Will automatically run build() the first time it is called.
        '''
        if self.b is not None:
            h = tf.matmul(inputs, self.w) + self.b
        else:
            h = tf.matmul(inputs, self.w)

        if activation is None:
            pass
        elif activation == 'relu':
            h = tf.nn.relu(features=h)
        else:
            sys.exit('Invalid \'activation\' argument: {}.'.format(activation))

        return h

def load_cora_data():
    data = dgl.data.citegrh.load_cora()
    features = tf.cast(data.features, dtype='float32')
    labels = tf.cast(data.labels, dtype='int32')
    train_mask = data.train_mask
    test_mask = data.test_mask
    g = dgl.from_networkx(data.graph)

    return g, features, labels, train_mask, test_mask
