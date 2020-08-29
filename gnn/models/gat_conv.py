import numpy as np
import dgl
import tensorflow as tf
from tensorflow.keras.models import Model
from gnn.models.tools import Linear



class GATLayer(Model):
    def __init__(self, out_feats, activation):
        super(GATLayer, self).__init__()

        self.activation = activation

        self.fc = Linear(units=out_feats, bias=False) # eq 1
        self.attention_fc = Linear(units = 1, bias=False) # eq 2

    def edge_attention(self, edges):
        # edge UDF for eq 2
        z_node_embeddings = tf.concat([edges.src['z'], edges.dst['z']], axis=1)
        attention_score = self.attention_fc(z_node_embeddings)

        # normalise scores to become probability/normalised attention distribution
        return {'e': tf.nn.leaky_relu(features=attention_score)}

    def message_func(self, edges):
        # message UDF for eq 3 & 4
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for eq 3 & 4
        alpha = tf.nn.softmax(logits=nodes.mailbox['e'], axis=1) # eq 3

        h = tf.math.reduce_sum(alpha * nodes.mailbox['z'], axis=1) # eq 4

        return {'h': h}

    def call(self, g, features):
        z = self.fc(features) # eq 1
        g.ndata['z'] = z

        g.apply_edges(self.edge_attention) # eq 2

        # eq 3 & 4
        g.update_all(message_func=self.message_func, reduce_func=self.reduce_func)
        return g.ndata.pop('h')
        

class MultiHeadGATLayer(Model):
    def __init__(self, out_feats, num_heads, activation, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        
        self.heads = []
        for i in range(num_heads):
            self.heads.append(GATLayer(out_feats, activation))

        self.merge = merge

    def call(self, g, features):
        head_outs = [attention_head(g, features) for attention_head in self.heads]

        # merge attention head outputs 
        if self.merge == 'cat':
            return tf.concat(head_outs, axis=1)
        elif self.merge == 'mean':
            return tf.math.reduce_mean(tf.stack(head_outs))

class GAT(Model):
    def __init__(self,
                 layers_config={'out_feats': [16, 7],
                                'activations': ['relu', None],
                                'num_heads': [2, 1]}):
        super(GAT, self).__init__()

        assert len(layers_config['out_feats']) >= 1, \
                'Must specify out_feats for >=1 layer(s)'
        assert len(layers_config['out_feats']) == len(layers_config['activations']) == len(layers_config['num_heads']), \
                'Must specify out_feats, activations and num_heads for all layers \
                (have specified {} out_feats, {} activations and {} num_heads)'.format(len(layers_config['out_feats']),
                                                                                       len(layers_config['activations']),
                                                                                       len(layers_config['num_heads']))
        assert layers_config['activations'][-1] is None, \
                'Final layer must have activation as None to output logits'
        assert layers_config['num_heads'][-1] == 1, \
                'Final GNN layer\'s attention layer must num_heads == 1'

        self._layers = []
        n_layers = len(layers_config['out_feats'])
        for i in range(n_layers):
            self._layers.append(MultiHeadGATLayer(out_feats=layers_config['out_feats'][i],
                                                  num_heads=layers_config['num_heads'][i],
                                                  activation=layers_config['activations'][i]))

    def call(self, g, features):
        h = features
        for layer in self._layers:
            h = layer(g, h)

        return h

                                                  

def evaluate(model, g, features, labels, mask):
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

    return acc
                                                  

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
                         'num_heads': [2, 1],
                         'activations': [None, None]}
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
                logits = model(g, features)
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













