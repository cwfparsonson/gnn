'''DGL message passing tutorial.

https://docs.dgl.ai/en/latest/tutorials/basics/3_pagerank.html

Tutorial uses dgl message passing to implement the PageRank algorithm

https://en.wikipedia.org/wiki/PageRank

PageRank works by counting the number and quality of links to a page to 
determine a rough estimate of how important the website is. The underlying 
assumption is that more important websites are likely to receive more links 
from other websites.

'''

import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
import dgl

path_figs = '../../data/figures/tutorials/message_passing/'

# init 100-node graph
N = 100 # num nodes
DAMP = 0.85 # damping factor
K = 10 # number of iterations
g = nx.nx.erdos_renyi_graph(N, 0.1)

# convert to dgl graph
g = dgl.from_networkx(g)
plt.figure()
nx.draw(g.to_networkx(), node_size=50, node_color=[[0.5,0.5,0.5,]])
plt.savefig(path_figs+'init_graph.png')

# init page rank value (pv) as 1/N for each node
g.ndata['pv'] = tf.cast(tf.ones(N)/N, dtype=tf.float32)

# store out-degree (number of edges out of node) as node feature
g.ndata['deg'] = tf.cast(g.out_degrees(g.nodes()), dtype=tf.float32)

def pagerank_message_func(edges):
    '''Generates a message for src node to pass to its dst node(s).

    'message functions' are defined on each edge in a DGL graph to generate
    a message by combining the edge feature with the features of its incident
    nodes

    In DGL, 'message functions' are 'Edge UDFs'

    'Edge UDFs' take a single argument 'edges'

    edges contains 3 members: (1) src (node whose features are being used
    to generate a message), (2) dst (node(s) which neighbour the src node and
    which will receive the generated message), and (3) data (the edge features)

    I.e. use src, dst and data to acces source node, destination node and edge
    features respectively.

    Want message being passed to destination nodes/neighbours to be the source
    node's page rank value divided by its out-degree


    '''
    message = edges.src['pv'] / edges.src['deg']
    
    return {'pv': message}

def pagerank_update_func(reduced_messages):
    '''Uses aggregated/reduced node messages to update a node's feature.

    'update functions' are defined on each node in a DGL graph to use the output
    of the node aggregation/reduce function to update the node's feature.

    '''
    # get updated node feature (the new pagerank value for the node)
    pv = (1 - DAMP) / N + DAMP * reduced_messages

    return pv

def pagerank_reduce_func(nodes):
    '''Aggregates node messages.

    'reduce functions' are defined on each node in a DGL graph to access the 
    node's mailbox, collect all messages in the mailbox and aggregate
    them using some aggregation/reduce function (e.g. sum, max, min etc.)

    In DGL, 'reduce functions' are 'Node UDFs'

    'Node UDFs' take a single argument 'nodes'

    'nodes' contains 2 members: (1) data (node features), and (2) mailbox (incoming
    message features stacked along the second dimension, hence axis=1 arg)

    Reduced messages can then be used to update the node feature using the
    update function.

    '''
    # collect messages from node's mailbox
    messages = nodes.mailbox['pv']

    # reduce/aggregate messages using reduce operation e.g. sum
    reduced_messages = tf.cast(tf.math.reduce_sum(messages, axis=1), dtype=tf.float32)

    # get updated node feature (new pagerank value for node)
    pv = pagerank_update_func(reduced_messages)

    return {'pv': pv}



def pagerank_level2(g):
    '''Implements the pagerank algorithm using user defined functions (UDFs).

    LEVEL 1 APPROACH (naive)
    A naive approach to performing the pagerank algorithm would be to iterate 
    over all the nodes one-by-one. 

    For each node:

    1. Send out messages from each node to all its neighbours (using message func)
    2. Receive the messages to compute new pagerank values (using reduce func)

    sudo code:
        # 1. Send out messages from src nodes u to dst nodes v
        # access src (u) dst (v) node pairs on each edge in graph to go through all nodes
        for u, v in zip(*g.edges()):
            g.send((u, v, message_func=pagerank_message_func))

        # 2. Receive messages at dst nodes v and use to update page rank value
        for v in g.nodes():
            g.recv(v, reduce_func=pagerank_reduce_func)

    LEVEL 2 APPROACH
    A better approach is to perform message passing simultaneously and then
    update the node features simultaneously (still using the same message passing
    and reduce fucntions). DGL can do this automatically. The easiest way
    is just to use the update_all function

    '''
    # perform 1. message passing and 2. node feature updates on all edges & nodes in dgl graph
    g.update_all(message_func=pagerank_message_func,
                 reduce_func=pagerank_reduce_func)



def pagerank_builtin(g):
    '''Implements pagerank algorithm using builtin message passing & reduce funcs.

    Some message and reduce functions are very common/standard in GNNs. Therefore,
    rather than creating our own UDFs, can use DGL's built-in functions.

    Using DGL builtin functions results in:
    
    1. Cleaner code
    2. DGL being able to automatically fuse operations together, resulting in 
    faster code

    '''
    # provide initial messages
    g.ndata['pv'] = g.ndata['pv'] / g.ndata['deg']

    # set message func and reduce func using builtin dgl funcs
    message_func = dgl.function.copy_src(src='pv', out='m')
    reduce_func = dgl.function.sum(msg='m', out='m_sum')

    # perform 1. message passing and 2. reduce messages
    g.update_all(message_func=message_func,
                 reduce_func=reduce_func)

    # use reduced messages to update nodes' feature
    g.ndata['pv'] = (1-DAMP) / N + DAMP * g.ndata['m_sum']
    




if __name__ == '__main__':
    import time

    # iteratively pass messages & update node features to perform pagerank algorithm

    # UDF implementation
    print('Performing UDF implementation...')
    udf_start = time.time()
    for k in range(K):
        pagerank_level2(g)
    udf_end = time.time()
    udf_runtime = udf_end - udf_start
    print('UDF run time: {}'.format(udf_runtime))    
    print('Page rank values for each node:\n{}'.format(g.ndata['pv']))



    # builtin implementation
    print('Performing builtin implementation...')
    bi_start = time.time()
    for k in range(K):
        pagerank_level2(g)
    bi_end = time.time()
    bi_runtime = bi_end - ui_start
    print('Builtin run time: {}'.format(bi_runtime))    
    print('Page rank values for each node:\n{}'.format(g.ndata['pv']))




    
