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
         
        self.units = units # num units in layer == num dims to output
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



def load_ppi_data():
    '''Protein-Protein Interaction network dataset.

    Contains 24 graphs (22 for train, 2 for test)

    Avrg nodes per graph: 2,372
    Features per node: 121
    Labels per node: 121

    '''
    data = dgl.data.PPIDataset()
    for g in data:
        features = tf.cast(g.ndata['feat'], dtype='float32')
        labels = tf.cast(g.ndata['label'], dtype='int32')
        train_mask = [True for _ in range(22)] + [False for _ in range(2)]
        test_mask = [False for _ in range(22)] + [True for _ in range(2)]

    graphs = data[0]

    return graphs, features, labels, train_mask, test_mask


def unpack_dataset(dataset):
    graph = dataset[0]
    features = graph.ndata['feat']
    labels = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']

    return graph, features, labels, train_mask, val_mask, test_mask

def load_data(dataset='cora'):
    if dataset == 'cora':
        data = dgl.data.CoraGraphDataset()
    elif dataset == 'citeseer':
        data = dgl.data.CiteseerGraphDataset()
    elif dataset == 'pubmed':
        data = dgl.data.PubmedGraphDataset()
    elif dataset == 'reddit':
        data = dgl.data.RedditDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))

    return unpack_dataset(data)















