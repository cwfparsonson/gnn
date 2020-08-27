import dgl
import tensorflow as tf
from tensorflow.keras.models import Model
from gnn.models.tools import Linear 



class GCNLayer(Model):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()

        # define the message & reduce function as usual
        self.message_func = dgl.function.copy_src(src='h', out='m')
        self.reduce_func = dgl.function.sum(msg='m', out='h')

        # add fully connected linear layer
        self.linear = Linear(units=out_feats)

    def call(self, g, feature):
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(self.message_func, self.reduce_func)
            h = g.ndata['h']

            return self.linear(h)

class GCN(Model):
    def __init__(self):
        super(GCN, self).__init__()

        self.layer1 = GCNLayer(in_feats=1433, out_feats=16)
        self.layer2 = GCNLayer(in_feats=16, out_feats=7)

    def call(self, g, features):
        # forward through 1st layer + activation func
        x = tf.nn.relu(features=(self.layer1(g, features)))

        # forward through second layer
        x = self.layer2(g, x)

        return x

def load_cora_data():
    data = dgl.data.citegrh.load_cora()
    features = tf.cast(data.features, dtype='float32')
    labels = tf.cast(data.labels, dtype='int32')
    train_mask = tf.cast(data.train_mask, dtype='int32')
    test_mask = tf.cast(data.test_mask, dtype='int32')
    g = dgl.from_networkx(data.graph)

    return g, features, labels, train_mask, test_mask


if __name__ == '__main__':
    import networkx
    import time
    import numpy as np

    # define gnn
    model = GCN()

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
            logp = tf.nn.log_softmax(logits, 1)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.gather(params=labels, indices=train_mask),
                                                           logits=tf.gather(params=logp, indices=train_mask))
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))

        print('Epoch: {} | Loss: {}'.format(epoch, tf.keras.backend.mean(loss)))

        



    

    








