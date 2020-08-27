import dgl
import tensorflow as tf
from tensorflow.keras.models import Model
from gnn.models.tools import Linear



class GATLayer(Model):
    def __init__(self, out_feats, activation):
        super(GATLayer, self).__init__()

        self.activation = activation

    def reset_parameters(self):
        pass

    def edge_attention(self, edges):
        pass

    def message_func(self, edges):
        pass

    def reduce_func(self, nodes):
        pass

    def call(self, g, features):
        pass

class MultiHeadGATLayer(Model):
    def __init__(self, out_feats, num_heads, activation, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        
        self.heads = []
        for i in range(num_heads):
            self.heads.append(GATLayer(out_feats, activation))

        self.merge = merge

    def call(self, g, features):
        pass

class GAT(Model):
    def __init__(self,
                 num_heads,
                 layers_config={'out_feats': [16, 7],
                                'activations': ['relu', None],
                                'num_heads': [4, 1]}):
        super(GAT, self).__init__()

        assert len(layers_config['out_feats']) >= 1, \
                'Must specify out_feats for >=1 layer(s)'
        assert len(layers_config['out_feats']) == len(layers_config['activations']), \
                'Must specify out_feats and activations for all layers \
                (have specified {} out_feats and {} activations)'.format(len(layers_config['out_feats']),
                                                                         len(layers_config['activations']))
        assert layers_config['activations'][-1] is None, \
                'Final layer must have activation as None to output logits'

        self._layers = []
        n_layers = len(layers_config['out_feats'])
        for i in range(n_layers):
            self._layers.append(MultiHeadGATLayer(out_feats=layers_config['out_feats'][i],
                                                  num_heads=layers_config['num_heads'][i],
                                                  activation=layers_config['activations'][i]))

    def call(self, g, features):
        pass

                                                  

                                                  














