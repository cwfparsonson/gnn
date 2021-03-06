import tensorflow as tf
import dgl
import math
from gnn.models.graph_conv import evaluate
from gnn.models.tools import load_data
from tensorboard.plugins.hparams import api as hp
import numpy as np
import os
import shutil
import copy
import json

class TensorboardWriter:
    def __init__(self, save_dir, hparams, overwrite=False, save_model=True):
        self.save_dir = save_dir
        self.init_dir = self.save_dir
        self.overwrite = overwrite
        self.save_model = save_model
        self.hparams = hparams

        self.METRIC_TEST_ACCURACY = 'test_accuracy'
        self.METRIC_TEST_ACCURACY_UNCERTAINTY = 'test_accuracy_uncertainty'
        self.METRIC_TRAINING_LOSS = 'training_loss'
        self.METRIC_VALIDATION_ACCURACY = 'validation_accuracy'

        self._init_dir()

    def _init_dir(self):
        try:
            os.mkdir(self.save_dir)
        except FileNotFoundError:
            print('Unable to create save_dir directory. Ensure that all but the last directory in the path already exist.')
            raise
        except FileExistsError:
            print('Directory already exists.')
            if self.overwrite:
                print('Overwriting directory.')
                shutil.rmtree(self.save_dir[:-1])
                os.mkdir(self.save_dir)
            else:
                print('Generating new directory.')
                v = 2
                new_path = self.save_dir
                while os.path.exists(new_path):
                    new_path = self.save_dir[:-1] + '_v' + str(v) + '/'
                    v += 1
                self.save_dir = new_path
                os.mkdir(self.save_dir)
                print('New directory:\n{}'.format(self.save_dir))

        assert self.save_dir[-1] == '/', \
                'Last character of self.save_dir must be \'/\', but is \'{}\''.format(self.save_dir[-1])

        with tf.summary.create_file_writer(self.save_dir).as_default():
            hp.hparams_config(hparams=self.hparams,
                              metrics=[hp.Metric(self.METRIC_TEST_ACCURACY, display_name='Test Accuracy'),
                                       hp.Metric(self.METRIC_TEST_ACCURACY_UNCERTAINTY, display_name='Test Accuracy Uncertainty'),
                                       hp.Metric(self.METRIC_TRAINING_LOSS, display_name='Training Loss'),
                                       hp.Metric(self.METRIC_VALIDATION_ACCURACY, display_name='Validation Accuracy')])


    def _train_test_model(self,
                          run_dir,
                          Model,
                          udf_hparams,
                          num_repeats=0):
        '''Runs a training loop given user-defined hyperparameters & performs test.

        Possible user defined hparam keys:

            - HP_LAYERS_CONFIG (name='layers_config') model-specific layers config dict
            - HP_DATASET (name='dataset') dataset to train, validate and test on
            - HP_OPTIMIZER (name='optimizer')
            - HP_LEARNING_RATE (name='learning_rate')
            - HP_NUM_EPOCHS (name='num_epochs')
            - HP_SHUFFLE (name='shuffle') # bool (whether or not to shuffle data)
            - HP_SAMPLE (name='sample') # bool (whether or not to sample)
            - HP_BATCH_SIZE (name='batch_size') # must have HP_SAMPLE==True to take effect
            - HP_NUM_NEIGHBOURS (name='num_neighbours') # no. neighbours to sample for each gnn layer https://docs.dgl.ai/en/0.5.x/api/python/dgl.dataloading.html#neighbor-sampler # must have HP_SAMPLE==True to take effect

        '''
        with tf.device('/cpu:0'):
            # define default hyperparameters (configured for basic GCN i.e. no attention heads etc)
            layers_config = {'out_feats': [16, 7],
                             'activations': ['relu', None],
                             'dropout_rates': [None, None],
                             'batch_norms': [False, False]}
            json_layers_config = json.dumps(layers_config) # conv to str
            HP_LAYERS_CONFIG = hp.HParam('layers_config', hp.Discrete(json_layers_config))
            HP_DATASET = hp.HParam('dataset', hp.Discrete(['cora']))
            HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam']))
            HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.0001]))
            HP_NUM_EPOCHS = hp.HParam('num_epochs', hp.Discrete([600]))
            HP_SHUFFLE = hp.HParam('shuffle', hp.Discrete([True]))
            HP_SAMPLE = hp.HParam('sample', hp.Discrete([True]))
            HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([35])) 
            HP_NUM_NEIGHBOURS = hp.HParam('num_neighbours', hp.Discrete([4])) # default to 0, which samples all neighbours for each hop/layer
            default_hparams = {HP_LAYERS_CONFIG: HP_LAYERS_CONFIG.domain.values[0],
                               HP_DATASET: HP_DATASET.domain.values[0],
                               HP_OPTIMIZER: HP_OPTIMIZER.domain.values[0],
                               HP_LEARNING_RATE: HP_LEARNING_RATE.domain.values[0],
                               HP_NUM_EPOCHS: HP_NUM_EPOCHS.domain.values[0],
                               HP_SHUFFLE: HP_SHUFFLE.domain.values[0],
                               HP_SAMPLE: HP_SAMPLE.domain.values[0],
                               HP_BATCH_SIZE: HP_BATCH_SIZE.domain.values[0],
                               HP_NUM_NEIGHBOURS: HP_NUM_NEIGHBOURS.domain.values[0],}

            # enter user defined hparams
            hparams = {}
            for key in default_hparams:
                if key.name not in [k.name for k in udf_hparams]:
                    hparams[key] = default_hparams[key]
                else:
                    udf_param = [udf_hparams[k] for k in udf_hparams if k.name == key.name]
                    hparams[key] = udf_param[0]

            # check all user defined params were entered
            for key in udf_hparams:
                if key.name not in [k.name for k in hparams]:
                    raise Exception('hparam key {} name is invalid. Accepted hparam key names:\n{}'.format(key, [k.name for k in default_hparams]))


            # reconfigure hyperparameters for this run
            for key in hparams:
                if key.name == 'dataset':
                    HP_DATASET = key
                elif key.name == 'layers_config':
                    HP_LAYERS_CONFIG = key
                elif key.name == 'optimizer':
                    HP_OPTIMIZER = key
                elif key.name == 'learning_rate':
                    HP_LEARNING_RATE = key
                elif key.name == 'num_epochs':
                    HP_NUM_EPOCHS = key
                elif key.name == 'shuffle':
                    HP_SHUFFLE = key
                elif key.name == 'sample':
                    HP_SAMPLE = key
                elif key.name == 'batch_size':
                    HP_BATCH_SIZE = key
                elif key.name == 'num_neighbours':
                    HP_NUM_NEIGHBOURS = key
                else:
                    raise Exception('Unrecognised hyperparameter defined in hparams.')

            if hparams[HP_SAMPLE]:
                mode = 'sampling'
            else:
                mode = 'no_sampling'
            hparams[HP_LAYERS_CONFIG] = json.loads(hparams[HP_LAYERS_CONFIG]) # convert from str to dict

            # load dataset
            dataset = hparams[HP_DATASET]
            g, features, labels, train_mask, val_mask, test_mask = load_data(dataset)

            # one-hot encode labels
            unique_labels = np.unique(labels)
            num_classes = len(unique_labels)
            onehot_labels = tf.one_hot(indices=labels, depth=num_classes)
            unique_onehot_labels = np.unique(onehot_labels, axis=0)
            label_to_onehot = {label: onehot for label, onehot in zip(unique_labels, unique_onehot_labels)}
            labels = tf.cast([label_to_onehot[l] for l in labels.numpy()], dtype=tf.int64) # convert to onehot

            # check final layer config
            if hparams[HP_LAYERS_CONFIG]['out_feats'][-1] == 'auto':
                # auto configure final layer num units to be num classes
                hparams[HP_LAYERS_CONFIG]['out_feats'][-1] = num_classes
            else:
                if num_classes != hparams[HP_LAYERS_CONFIG]['out_feats'][-1]:
                    raise Exception('Set number of units in final layer of model ({}) to number of possible classes to classify ({}).'.format(hparams[HP_LAYERS_CONFIG]['out_feats'][-1], num_classes))


            # add edges between each node and itself to preserve old node representations
            g.add_edges(g.nodes(), g.nodes())

            # define optimiser
            if hparams[HP_OPTIMIZER] == 'adam':
                opt = tf.keras.optimizers.Adam(learning_rate=hparams[HP_LEARNING_RATE])
            else:
                opt = tf.keras.optimizers.SGD(learning_rate=hparams[HP_LEARNING_RATE])

            # init run trackers
            all_test_accuracy = []
            all_training_loss = {epoch: [] for epoch in range(hparams[HP_NUM_EPOCHS])}
            all_validation_accuracy = {epoch: [] for epoch in range(hparams[HP_NUM_EPOCHS])}

            # get node ids
            node_ids = g.nodes()
            index = 0
            train_nids, val_nids = [], []
            for tm, vm in zip(train_mask, val_mask):
                if tm and vm:
                    raise Exception('Sample registered as both train and validation sample, should only be one.')
                elif tm:
                    train_nids.append(node_ids[index])
                elif vm:
                    val_nids.append(node_ids[index])
                else:
                    # is test mask
                    pass
                index += 1

            if hparams[HP_SHUFFLE]:
                orig_train_nids = copy.deepcopy(train_mask)

            # repeat training loops to get uncertainty
            for _ in range(num_repeats+1):
                # init model
                model = Model(layers_config=hparams[HP_LAYERS_CONFIG])

                # begin training loop
                for epoch in range(hparams[HP_NUM_EPOCHS]):
                    epoch_loss = []

                    # shuffle
                    if hparams[HP_SHUFFLE]:
                        train_nids = tf.random.shuffle(train_nids)
                    else:
                        pass

                    # train for this epoch
                    if mode == 'sampling':
                        # sample minibatches
                        if hparams[HP_NUM_NEIGHBOURS] != 0:
                            fanouts = [hparams[HP_NUM_NEIGHBOURS] for _ in range(model.num_layers)] # sampling fixed num neighbour nodes each layer
                        else:
                            fanouts = [None for _ in range(model.num_layers)] # sample all neighbours
                        num_batches = math.ceil(int(len(train_nids)/hparams[HP_BATCH_SIZE]))
                        train_mini_batches = []
                        prev_index = 0
                        index = copy.deepcopy(hparams[HP_BATCH_SIZE])
                        for batch_num in range(num_batches):
                            train_mini_batches.append(train_nids[prev_index:index])
                            prev_index = copy.deepcopy(index)
                            index += hparams[HP_BATCH_SIZE]

                        sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts=fanouts, replace=False)
                        collator = dgl.dataloading.NodeCollator(g, train_nids, sampler)
                        
                        # train with sampling and minibatching
                        for batch_nodes in train_mini_batches:
                            with tf.GradientTape() as tape:
                                input_nodes, output_nodes, blocks = collator.collate(batch_nodes)
                                with tf.device('/gpu:0'): # uncomment & unindent below if training on cpu
                                    input_features = blocks[0].srcdata['feat']
                                    output_labels = blocks[-1].dstdata['label']
                                    onehot_output_labels = tf.cast([label_to_onehot[l] for l in output_labels.numpy()],dtype=tf.int64)
                                    blocks = [block.to('/gpu:0') for block in blocks] # uncomment if training on cpu
                                    logits = model(blocks, input_features, training=True, mode=mode)
                                    loss = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_output_labels, logits=logits)
                                    epoch_loss.append(tf.math.reduce_mean(loss))
                                    grads = tape.gradient(loss, model.trainable_variables)
                                    opt.apply_gradients(zip(grads, model.trainable_variables))
                    else:
                        # train without sampling or minibatching
                        with tf.GradientTape() as tape:
                            logits = model(g, features, training=True, mode=mode)
                            output_logits = tf.gather(logits, train_nids)
                            output_labels = tf.gather(labels, train_nids)
                            loss = tf.nn.softmax_cross_entropy_with_logits(labels=output_labels,logits=output_logits)
                            epoch_loss.append(tf.math.reduce_mean(loss))
                            grads = tape.gradient(loss, model.trainable_variables)
                            opt.apply_gradients(zip(grads, model.trainable_variables))

                    # validation testing
                    epoch_acc = evaluate(model, g, features, labels, val_nids)

                    # summarise epoch
                    epoch_loss = np.mean(np.array(epoch_loss))
                    all_training_loss[epoch].append(epoch_loss)
                    all_validation_accuracy[epoch].append(tf.math.reduce_mean(epoch_acc))

                # test fully trained model
                acc = evaluate(model, g, features, labels, test_mask)
                all_test_accuracy.append(acc)

            # summarise run
            # training loss and validation
            for epoch in range(hparams[HP_NUM_EPOCHS]):
                loss = np.mean(np.array(all_training_loss[epoch]))
                acc = np.mean(np.array(all_validation_accuracy[epoch]))
                with tf.summary.create_file_writer(run_dir).as_default():
                    tf.summary.scalar(self.METRIC_TRAINING_LOSS, data=loss, step=epoch)
                    tf.summary.scalar(self.METRIC_VALIDATION_ACCURACY, data=acc, step=epoch)
            # test
            mean_accuracy = np.mean(np.array(all_test_accuracy))
            if num_repeats > 1:
                uncertainty = (np.max(np.array(all_test_accuracy)) - np.min(np.array(all_test_accuracy))) / 2
            else:
                uncertainty = 0

            return model, mean_accuracy, uncertainty 


    def run(self, run_name, Model, hparams, num_repeats=1):
        '''Trains and tests model with given hparams and tracks with tensorboard.

        Args:
            run_name (str): Name of run (will be saved in self.save_dir in a 
                folder under this name).
            Model (obj): GNN model to train and test (N.B. Must be uninitialised)
            hparams (dict): User-defined hyperparameters to use.
            num_repeats (int): Number of times to repeat training and testing
                (will return the average accuracy and the uncertainty in the
                accuracy value).

        '''
        run_dir = self.save_dir + run_name
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams) # record hparams used in this run
            model, accuracy, uncertainty = self._train_test_model(run_dir, 
                                                                  Model,
                                                                  hparams, 
                                                                  num_repeats=num_repeats)
        if self.save_model:
            model.save_weights(run_dir + '/trained_model_weights', overwrite=True, save_format='tf')
        with tf.summary.create_file_writer(run_dir).as_default():
            tf.summary.scalar(self.METRIC_TEST_ACCURACY, data=accuracy, step=1)
            tf.summary.scalar(self.METRIC_TEST_ACCURACY_UNCERTAINTY, data=uncertainty, step=1)




