import numpy as np
import dgl
import tensorflow as tf
from tensorflow.keras.layers import Layer
import sys

class Linear(Layer):
    '''Fully-connected linear layer.'''

    def __init__(self, 
                 units=32,
                 bias=True,
                 batch_norm=False,
                 dropout_rate=None):
        super(Linear, self).__init__()
         
        self.units = units # num units in layer == num dims to output
        self.bias = bias # whether or not to use bias units in layer
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate

        if dropout_rate is not None and batch_norm:
            print('WARNING: Have set to use both dropout and batch normalisation. \
                    Research has shown that using both can lead to poorer results \
                    than using neither. It is generally recommended to only use \
                    batch normalisation. See https://arxiv.org/abs/1801.05134.')

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

        # init batch norm layer with learnable params
        self.bn_layer = tf.keras.layers.BatchNormalization()


    def call(self, inputs, training, activation=None):
        '''Performs forward propagation through layer when layer is called.

        Will automatically run build() the first time it is called.

        Uses order (ongoing debate about which order is best):

            conv/fully_connected (linear op) -> activation (non-linear op) -> dropout -> batch_norm

        '''
        # linear operation
        if self.b is not None:
            h = tf.matmul(inputs, self.w) + self.b
        else:
            h = tf.matmul(inputs, self.w)

        # activation (non-linear operation)
        if activation is None:
            pass
        elif activation == 'relu':
            h = tf.nn.relu(features=h)
        else:
            raise Exception('Invalid \'activation\' argument: {}.'.format(activation))

        # dropout units
        if self.dropout_rate is not None and training:
            h = tf.nn.dropout(h, self.dropout_rate)
        else:
            pass

        # normalise batch
        if self.batch_norm:
            h = self.bn_layer(h, training=training)

        return h


def evaluate(model, g, features, labels, mask):
    logits = model(g, features, training=False, mode='no_sampling')
    if type(mask[0].numpy()) == np.bool_:
        logits = tf.boolean_mask(tensor=logits, mask=mask)
        labels = tf.boolean_mask(tensor=labels, mask=mask)
    else:
        logits = tf.gather(logits, mask)
        labels = tf.gather(labels, mask)
    indices = tf.math.argmax(logits, axis=1)
    indices = tf.one_hot(indices=indices, depth=len(labels[0]))
    correct = 0
    for i in range(len(labels)):
        if np.array_equal(indices.numpy()[i], labels.numpy()[i]):
            correct+=1
    acc = correct / len(labels)

    return acc


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















