import dgl
import tensorflow as tf
from tensorflow.keras.models import Model
from gnn.models.tools import Linear 



class GCNLayer(Model):
    def __init__(self, out_feats, activation):
        super(GCNLayer, self).__init__()

        self.activation = activation

        # define the message & reduce function as usual
        self.message_func = dgl.function.copy_src(src='h', out='m')
        self.reduce_func = dgl.function.sum(msg='m', out='h')

        # add fully connected linear layer
        self.linear = Linear(units=out_feats)

    def call(self, g, feature):
        with g.local_scope():
            # pass messages between nodes & reduce messages
            g.ndata['h'] = feature
            g.update_all(self.message_func, self.reduce_func)
            h = g.ndata['h']

            # forward reduced messages through NN layer
            h = self.linear(h, self.activation)

            return h

# class GCN(Model):
    # def __init__(self):
        # super(GCN, self).__init__()

        # self.layer1 = GCNLayer(in_feats=1433, out_feats=16)
        # self.layer2 = GCNLayer(in_feats=16, out_feats=7)

    # def call(self, g, features):
        # # forward through 1st layer + activation func
        # x = self.layer1(g, features, activation='relu')

        # # forward through second layer (but not through activation func)
        # x = self.layer2(g, x, activation=None)

        # return x


class GCN(Model):
    '''Implementation of convolutional graph neural network.

    Must configure input, hidden and output layers by using the layers_config
    dict. Specify out_feats for each layer to specify the number of output
    features each layer should output. The model will automatically set the
    number of input features for each layer based on what is being passed into
    the layer when called. N.B. NN must have minimum of 1 layer.

    The final layer (the output layer) should have activation==None so that
    the model outputs logits, which may then be externally converted into 
    probability predictions using e.g. softmax.

    N.B. The final value in out_feats arg list should be the number of classes
    being classified by the output layer (e.g. if have 7 classes to classify 
    inputs into, final out_feats val should be 7)
    '''
    def __init__(self,
                 layers_config={'out_feats': [16, 7],
                                'activations': ['relu', None]}):
        super(GCN, self).__init__()

        assert len(layers_config['out_feats']) >= 1, \
                'Must specify out_feats for >=1 layer(s)'
        assert len(layers_config['out_feats']) == len(layers_config['activations']), \
                'Must specify out_feats and activations for all layers \
                (have specified {} out_feats and {} activations)'.format(len(layers_config['out_feats']),
                                                                         len(layers_config['activations']))
        assert layers_config['activations'][-1] is None, \
                'Final layer must have activation as None to output logits'

        # define layers
        self._layers = []
        n_layers = len(layers_config['out_feats'])
        for i in range(n_layers):
            self._layers.append(GCNLayer(out_feats=layers_config['out_feats'][i],
                                        activation=layers_config['activations'][i]))

    def call(self, g, features):
        '''Forward graph node features through GNN, generating hidden feature representations.'''
        h = features
        for layer in self._layers:
            h = layer(g, h)

        return h

def evaluate(model, features, labels, mask):
    logits = model(g, features)
    logits = tf.boolean_mask(tensor=logits, mask=mask)
    labels = tf.boolean_mask(tensor=labels, mask=mask)
    indices = tf.math.argmax(logits, axis=1)
    indices = tf.one_hot(indices=indices, depth=len(labels[0]))
    correct = 0
    for i in range(len(labels)):
        if np.array_equal(indices.numpy()[i], labels.numpy()[i]):
            correct+=1
    acc = correct / len(labels)
    # acc = tf.reduce_mean(tf.cast(indices==labels, dtype=tf.float32))

    return acc



if __name__ == '__main__':
    import networkx
    import numpy as np
    from gnn.models.tools import load_cora_data

    # define gnn model
    layers_config = {'out_feats': [16, 7],
                     'activations': ['relu', None]}
    model = GCN(layers_config=layers_config)

    # load cora dataset
    g, features, labels, train_mask, test_mask = load_cora_data()

    # one-hot encode labels
    num_classes = int(len(np.unique(labels)))
    labels = tf.one_hot(indices=labels, depth=num_classes)

    # add edges between each node and itself to preserve old node representations
    g.add_edges(g.nodes(), g.nodes())

    # define optimiser
    opt = tf.keras.optimizers.Adam(learning_rate=1e-2)

    # run training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            logits = model(g, features)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.boolean_mask(tensor=labels, mask=train_mask),
                                                           logits=tf.boolean_mask(tensor=logits, mask=train_mask))
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
        acc = evaluate(model, features, labels, test_mask)
        print('Epoch: {} | Loss: {} | Accuracy: {}'.format(epoch, tf.keras.backend.mean(loss), acc))


        


    
