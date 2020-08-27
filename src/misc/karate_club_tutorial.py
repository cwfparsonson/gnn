'''Implementation of DGL beginner tutorial.

https://docs.dgl.ai/tutorials/basics/1_first.html

The tutorial is based on the “Zachary’s karate club” problem. The karate club 
is a social network that includes 34 members and documents pairwise links 
between members who interact outside the club. The club later divides into two 
communities led by the instructor (node 0) and the club president (node 33). 
The network is visualized as follows with the color indicating the community.

The task is to predict which side (0 or 33) each member tends to join given the 
social network itself.
'''

import dgl
from dgl.nn import GraphConv
import numpy as np
import networkx as nx
import tensorflow as tf
from tensorflow.keras.models import Model

def build_karate_club_graph():
    # All 78 edges are stored in two numpy arrays. One for source endpoints
    # while the other for destination endpoints.
    src = np.array([1, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 9, 10, 10,
        10, 11, 12, 12, 13, 13, 13, 13, 16, 16, 17, 17, 19, 19, 21, 21,
        25, 25, 27, 27, 27, 28, 29, 29, 30, 30, 31, 31, 31, 31, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33,
        33, 33, 33, 33, 33, 33, 33, 33, 33, 33])
    dst = np.array([0, 0, 1, 0, 1, 2, 0, 0, 0, 4, 5, 0, 1, 2, 3, 0, 2, 2, 0, 4,
        5, 0, 0, 3, 0, 1, 2, 3, 5, 6, 0, 1, 0, 1, 0, 1, 23, 24, 2, 23,
        24, 2, 23, 26, 1, 8, 0, 24, 25, 28, 2, 8, 14, 15, 18, 20, 22, 23,
        29, 30, 31, 8, 9, 13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30,
        31, 32])
    # Edges are directional in DGL; Make them bi-directional.
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    # Construct a DGLGraph
    return dgl.DGLGraph((u, v))

class GCN(Model):
    '''Simple implementation of 2-layer Graph Convolutional Network (GCN).

    Args:
        in_feats: dimension/number of features of input data to first layer
        hidden_size: dimension/number of features of data output by first
            layer and input into second layer
        num_classes: number of classes to make predictions for (is dimensions
            of output of second layer)

    '''
    def __init__(self, in_feats, hidden_size, num_classes):
        # inherit from backend NN model
        super(GCN, self).__init__()

        # add 2 conv layers
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)

    def call(self, g, inputs):
        '''Implement GCN's forward pass through NN.'''
        h = self.conv1(g, inputs)
        h = tf.nn.relu(h)
        h = self.conv2(g, h)

        return h


# def loss(model, loss_fn, G, x, y):
      # y_ = model(G, x)

      # return loss_fn(y_true=y, y_pred=y_)

# def grad(model, G, loss_fn, inputs, targets):
      # with tf.GradientTape() as tape:
          # loss_value = loss(model, loss_fn, G, inputs, targets)

      # return loss_value, tape.gradient(loss_value, model.trainable_variables)





if __name__ == '__main__':
    import matplotlib.pyplot as plt

    path_figs = '../data/figures/karate_club/'

    # 1. Create DGL graph
    G = build_karate_club_graph()

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    print('Num nodes: {}, num edges: {}'.format(num_nodes, num_edges)) 

    # can convert to networkx graph
    nx_G = G.to_networkx().to_undirected()
    plt.figure()
    pos = nx.kamada_kawai_layout(nx_G)
    nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
    plt.savefig(path_figs + 'karate_club_graph.png')

    # 2. Assign features to nodes/edges i.e. node features are learnable embeddings
    # use embedding with 5 dimensions
    embed = tf.keras.layers.Embedding(input_dim=num_nodes,
                                      output_dim=5)
    embeddings = embed(np.array(list(nx_G.nodes)))

    # assign feature values
    G.ndata['feat'] = embeddings
    print('Node 2 embedding:\n{}'.format(G.ndata['feat'][2]))

    # 3. Define a GNN
    model = GCN(in_feats=5, hidden_size=5, num_classes=2)

    # 4. Prepare data
    inputs = embeddings
    labeled_nodes = tf.constant([0, 33], dtype=tf.float32) # only instructor and president are labelled
    # labels = tf.constant([0, 1], dtype=tf.float32) # binary classification
    labels = tf.one_hot(indices=[0,1], depth=2) # one-hot encode binary classificaion problem so have same dimension logits/predictions as labels

    # 5. Train GNN
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    logits = model(G, inputs).numpy()
    print('init model output logits:\n{}'.format(logits))
    probs = tf.nn.sigmoid(logits).numpy()
    print('prediction probabilities:\n{}'.format(probs))

    num_epochs = 50
    all_logits = []
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:

            # get logit predictions
            logits = model(G, inputs)
            all_logits.append(logits)
            # predictions = tf.argmax(logits, axis=1)
            # print('predictions:\n{}'.format(predictions))

            # convert logit predictions to prediction probabilities
            prediction_probabilites = tf.nn.sigmoid(logits)

            # only evaluate loss of predictions we have labels for
            labeled_prediction_probabilities = [prediction_probabilites[int(i)] for i in labeled_nodes]
            loss = loss_fn(y_true=labels, y_pred=labeled_prediction_probabilities)

            # calculate gradients and update NN parameters to minimise loss
            grads = tape.gradient(loss,model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))

        print('Epoch: {} | Loss: {}'.format(epoch, loss))



    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    def draw(i):
        cls1color = '#00FFFF'
        cls2color = '#FF00FF'
        pos = {}
        colors = []
        for v in range(34):
            pos[v] = all_logits[i][v].numpy()
            cls = pos[v].argmax()
            colors.append(cls1color if cls else cls2color)
        ax.cla()
        ax.axis('off')
        ax.set_title('Epoch: %d' % i)
        nx.draw_networkx(nx_G.to_undirected(), pos, node_color=colors,
                with_labels=True, node_size=300, ax=ax)

    fig = plt.figure(dpi=150)
    fig.clf()
    ax = fig.subplots()
    draw(0)  # draw the prediction of the first epoch
    plt.savefig(path_figs + 'init_predictions.png')
    plt.close()

    ani = animation.FuncAnimation(fig, draw, frames=len(all_logits), interval=200)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(path_figs + 'animation.mp4', writer=writer)

    for epoch in range(len(all_logits)):
        logits = all_logits[epoch]
        predictions = tf.argmax(logits, axis=1)
        # print('predictions:\n{}'.format(predictions))
            
        









