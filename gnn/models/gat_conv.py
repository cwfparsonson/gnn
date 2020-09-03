import dgl
import tensorflow as tf
from tensorflow.keras.models import Model
from gnn.models.tools import Linear, evaluate



class GATLayer(Model):
    def __init__(self, out_feats, batch_norm=False, dropout_rate=None):
        super(GATLayer, self).__init__()

        # init fully connected linear layer
        self.fc = Linear(units=out_feats, bias=False, activation=None, batch_norm=batch_norm, dropout_rate=dropout_rate) # eq 1

        # init fully connected attention layer
        self.attention_fc = Linear(units=1, bias=False, activation='leaky_relu', batch_norm=batch_norm, dropout_rate=dropout_rate) # eq 2

    def edge_attention(self, edges):
        # edge UDF for eq 2
        z_node_embeddings = tf.concat([edges.src['z'], edges.dst['z']], axis=1)

        # get normalised attention distribution using an activation func
        attention_score = self.attention_fc(z_node_embeddings, training=False)

        return {'e': attention_score}

    def gat_message_func(self, edges):
        # message UDF for eq 3 & 4
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def gat_reduce_func(self, nodes):
        # reduce UDF for eq 3 & 4
        alpha = tf.nn.softmax(logits=nodes.mailbox['e'], axis=1) # eq 3

        h = tf.math.reduce_sum(alpha * nodes.mailbox['z'], axis=1) # eq 4

        return {'h': h}

    def call(self, g, features, training, mode):

        if mode == 'no_sampling':
            with g.local_scope():
                z = self.fc(inputs=features, training=training) # eq 1
                g.ndata['z'] = z

                g.apply_edges(self.edge_attention) # eq 2

                # eq 3 & 4
                g.update_all(message_func=self.gat_message_func, reduce_func=self.gat_reduce_func)
                return g.ndata.pop('h')

        elif mode == 'sampling':
            block = g
            with block.local_scope():
                z = self.fc(inputs=features, training=training) # eq 1
                z_src = z
                z_dst = z[:block.number_of_dst_nodes()]
                block.srcdata['z'] = z_src
                block.dstdata['z'] = z_dst

                block.apply_edges(self.edge_attention) # eq 2

                # pass messages between nodes & reduce/aggregate message
                block.update_all(message_func=self.gat_message_func, reduce_func=self.gat_reduce_func)
                return block.dstdata.pop('h')
        

class MultiHeadGATLayer(Model):
    def __init__(self, out_feats, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        
        self.heads = []
        for i in range(num_heads):
            self.heads.append(GATLayer(out_feats))

        self.merge = merge

    def call(self, g, features, training, mode):
        head_outs = [attention_head(g, features, training=training, mode=mode) for attention_head in self.heads]

        # merge attention head outputs 
        if self.merge == 'cat':
            return tf.concat(head_outs, axis=1)
        elif self.merge == 'mean':
            return tf.math.reduce_mean(tf.stack(head_outs))

class GAT(Model):
    def __init__(self,
                 layers_config={'out_feats': [16, 7],
                                'num_heads': [2, 1]}):
        super(GAT, self).__init__()

        self.model_name = 'gat_conv'

        # check layers config is valid
        num_layers = len(layers_config['out_feats'])
        num_heads = len(layers_config['num_heads'])
        if num_layers > num_heads:
            print('Warning: Only specified {} num_heads for {} layer model (num_layers defined by number of elements in \'out_feats\'). Appending first element of \'num_heads\' to remaining layers (except final output layer).'.format(num_heads, num_layers))
            while len(layers_config['num_heads']) < num_layers:
                layers_config['num_heads'].insert(0, layers_config['num_heads'][0])
        elif num_layers < num_heads:
            print('Warning: Specified {} num_heads for {} layer model (num_layers defined by number of elements in \'out_feats\'). Removing extra elements of \'num_heads\' starting from first entry (except final output layer).'.format(num_heads, num_layers))
            while len(layers_config['num_heads']) > num_layers:
                del layers_config['num_heads'][0]
        else:
            pass

        assert len(layers_config['out_feats']) >= 1, \
                'Must specify out_feats for >=1 layer(s)'
        assert layers_config['num_heads'][-1] == 1, \
                'Final GNN layer\'s attention layer must have num_heads == 1'

        self._layers = []
        self.num_layers = len(layers_config['out_feats'])
        for i in range(self.num_layers):
            self._layers.append(MultiHeadGATLayer(out_feats=layers_config['out_feats'][i],
                                                  num_heads=layers_config['num_heads'][i]))

    def call(self, g, features, training, mode='no_sampling'):
        h = features
        if mode == 'no_sampling':
            for layer in self._layers:
                h = layer(g, h, training=training, mode=mode)
        elif mode == 'sampling':
            blocks = iter(g)
            for layer in self._layers:
                h = layer(next(blocks), h, training=training, mode=mode)

        return h

                                                  

                                                  

if __name__ == '__main__':
    import numpy as np
    from gnn.models.tools import load_data
    import matplotlib.pyplot as plt

    dataset = 'cora'
    path_figures = '../../data/figures/'+str(dataset)+'/gat_conv/'

    print('Available devices:\n{}'.format(tf.config.list_physical_devices()))
    device = '/:CPU:0'
    with tf.device(device):

        # load dataset
        g, features, labels, train_mask, val_mask, test_mask = load_data(dataset)

        # one-hot encode labels
        num_classes = int(len(np.unique(labels)))
        labels = tf.one_hot(indices=labels, depth=num_classes)

        # define gnn model
        layers_config = {'out_feats': [8, num_classes],
                         'num_heads': [2, 1]}
        model = GAT(layers_config=layers_config)

        # add edges between each node and itself to preserve old node representations
        g.add_edges(g.nodes(), g.nodes())

        # define optimiser
        opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

        # run training loop
        all_loss = []
        all_acc = []
        all_epochs = []
        num_epochs = 200
        for epoch in range(num_epochs):
            with tf.GradientTape() as tape:
                logits = model(g, features, training=True, mode='no_sampling')
                loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.boolean_mask(tensor=labels, mask=train_mask),
                                                               logits=tf.boolean_mask(tensor=logits, mask=train_mask))
                all_loss.append(tf.keras.backend.mean(loss))
                grads = tape.gradient(loss, model.trainable_variables)
                opt.apply_gradients(zip(grads, model.trainable_variables))
            acc = evaluate(model, g, features, labels, val_mask)
            all_acc.append(acc)
            all_epochs.append(epoch)
            print('Epoch: {} | Training loss: {} | Validation accuracy: {}'.format(epoch, tf.keras.backend.mean(loss), acc))

        acc = evaluate(model, g, features, labels, test_mask)
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













