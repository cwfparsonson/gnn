import dgl
import tensorflow as tf
from tensorflow.keras.models import Model
from gnn.models.tools import Linear, evaluate



class GCNLayer(Model):
    def __init__(self, out_feats, activation, batch_norm=False, dropout_rate=None):
        super(GCNLayer, self).__init__()

        self.activation = activation

        # define the message & reduce function as usual
        self.gcn_message_func = dgl.function.copy_src(src='h', out='m')
        self.gcn_reduce_func = dgl.function.sum(msg='m', out='h')

        # add fully connected linear layer
        self.linear = Linear(units=out_feats, batch_norm=batch_norm, dropout_rate=dropout_rate)

    def call(self, g, features, training, mode):

        if mode == 'no_sampling':
            with g.local_scope():
                # enter node features
                g.ndata['h'] = features 

                # pass messages between nodes & reduce/aggregate messages
                g.update_all(self.gcn_message_func, self.gcn_reduce_func)
                h = g.ndata['h']

                # forward aggregated messages through NN layer -> GNN node representation
                h = self.linear(inputs=h, training=training, activation=self.activation)

                return h

        elif mode == 'sampling':
            block = g
            with block.local_scope():
                # enter node features
                h_src = features
                h_dst = features[:block.number_of_dst_nodes()]
                block.srcdata['h'] = h_src
                block.dstdata['h'] = h_dst
                
                # pass messages between nodes & reduce/aggregate message
                block.update_all(self.gcn_message_func, self.gcn_reduce_func)
                h = block.dstdata['h']

                # forward aggregated messages through NN layer -> GNN node representation
                h = self.linear(inputs=h, training=training, activation=self.activation)

                return h
                



class GCN(Model):
    '''Implementation of convolutional graph neural network.

    Must configure input, hidden and output layers by using the layers_config
    dict. Specify out_feats for each layer to specify the number of output
    features each layer should output. The model will automatically set the
    number of input features for each layer based on what is being passed into
    the layer when called. N.B. NN must have minimum of 1 layer. N.B.2. The number
    of elements given to out_feats and activations lists should be equal and
    will correspond to the total number of layers (input, hidden and output)
    in the model.

    The final layer (the output layer) should have activation==None so that
    the model outputs logits, which may then be externally converted into 
    probability predictions using e.g. softmax.

    N.B. The final value in out_feats arg list should be the number of classes
    being classified by the output layer (e.g. if have 7 classes to classify 
    inputs into, final out_feats val should be 7)
    '''
    def __init__(self,
                 layers_config={'out_feats': [16, 7],
                                'activations': ['relu', None],
                                'batch_norms': [False, False],
                                'dropout_rates': [0.5, None]}):
        super(GCN, self).__init__()

        assert len(layers_config['out_feats']) >= 1, \
                'Must specify out_feats for >=1 layer(s)'
        assert len(layers_config['out_feats']) == len(layers_config['activations']), \
                'Must specify out_feats and activations for all layers \
                (have specified {} out_feats and {} activations)'.format(len(layers_config['out_feats']),
                                                                         len(layers_config['activations']))
        assert layers_config['activations'][-1] is None, \
                'Final layer must have activation as None to output logits'
        assert layers_config['dropout_rates'][-1] is None, \
                'Final layer must have dropout_rate as None to avoid dropping predictions.'

        # define layers
        self._layers = []
        n_layers = len(layers_config['out_feats'])
        for i in range(n_layers):
            self._layers.append(GCNLayer(out_feats=layers_config['out_feats'][i],
                                         activation=layers_config['activations'][i],
                                         batch_norm=layers_config['batch_norms'][i],
                                         dropout_rate=layers_config['dropout_rates'][i]))

        self.num_layers = len(self._layers)

    def call(self, g, features, training, mode='no_sampling'):
        '''Forward graph node features through GNN, generating hidden feature representations.'''
        if mode == 'sampling' and len(g) != len(self._layers):
            raise Exception('Number of blocks must ({}) must == number of GNN layers ({}).'.format(len(g),len(self._layers)))
        if type(g) == list and mode == 'no_sampling':
            raise Exception('Specified mode as no_sampling but providing list of blocks. Set mode=\'sampling\' or provide whole graph.')
        elif type(g) != list and mode == 'sampling':
            raise Exception('Specified mode as sampling but not providing list of blocks. Set mode=\'no_sampling\' or provide list of blocks.')

        h = features
        if mode == 'no_sampling':
            for layer in self._layers:
                h = layer(g, h, training=training, mode=mode)
        elif mode == 'sampling':
            blocks = iter(g)
            for layer in self._layers:
                h = layer(next(blocks), h, training=training, mode=mode)
        else:
            raise Exception('Unrecognised mode: {}'.format(mode))

        return h




if __name__ == '__main__':
    import numpy as np
    from gnn.models.tools import load_data
    import matplotlib.pyplot as plt

    dataset = 'cora'
    path_figures = '../../data/figures/'+str(dataset)+'/graph_conv/'

    print('Available devices:\n{}'.format(tf.config.list_physical_devices()))
    device = '/:CPU:0'
    with tf.device(device):
        # load dataset
        g, features, labels, train_mask, val_mask, test_mask = load_data(dataset)

        # one-hot encode labels
        num_classes = int(len(np.unique(labels)))
        labels = tf.one_hot(indices=labels, depth=num_classes)

        # define gnn model
        layers_config = {'out_feats': [16, num_classes],
                         'activations': ['relu', None],
                         'dropout_rates': [0.5, None]}
        model = GCN(layers_config=layers_config)

        # add edges between each node and itself to preserve old node representations
        g.add_edges(g.nodes(), g.nodes())

        # define optimiser
        opt = tf.keras.optimizers.Adam(learning_rate=1e-2)

        # run training loop
        all_loss = []
        all_acc = []
        all_epochs = []
        all_logits = []
        num_epochs = 50
        for epoch in range(num_epochs):
            with tf.GradientTape() as tape:
                logits = model(g, features)
                all_logits.append(logits)
                loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.boolean_mask(tensor=labels, mask=train_mask),
                                                               logits=tf.boolean_mask(tensor=logits, mask=train_mask))
                all_loss.append(tf.keras.backend.mean(loss))
                grads = tape.gradient(loss, model.trainable_variables)
                opt.apply_gradients(zip(grads, model.trainable_variables))
            acc = evaluate(model, g, features, labels, val_mask)
            all_acc.append(acc)
            all_epochs.append(epoch)
            print('Epoch: {} | Training loss: {} | Validation accuracy: {}'.format(epoch, tf.keras.backend.mean(loss), acc))

        acc = evaluate(model, features, labels, test_mask)
        print('Final test accuracy: {}'.format(acc))

        plt.figure()
        plt.scatter(all_epochs, all_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.savefig(path_figures + 'loss.png')

        plt.figure()
        plt.scatter(all_epochs, all_acc)
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.savefig(path_figures + 'accuracy.png')
            

        
