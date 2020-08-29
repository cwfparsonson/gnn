import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import numpy as np
import os
import shutil
import copy

class TensorboardWriter:
    def __init__(self, logs_dir, hparams, overwrite=False):
        self.logs_dir = logs_dir
        self.overwrite = overwrite
        self.hparams = hparams

        self.METRIC_TEST_ACCURACY = 'test_accuracy'
        self.METRIC_TEST_ACCURACY_UNCERTAINTY = 'test_accuracy_uncertainty'
        self.METRIC_TRAINING_LOSS = 'training_loss'
        self.METRIC_VALIDATION_ACCURACY = 'validation_accuracy'

        self._init_dir()

    def _init_dir(self):
        try:
            os.mkdir(self.logs_dir)
        except FileNotFoundError:
            print('Unable to create save_weight directory. Ensure that all but the last directory in the path already exist.')
            raise
        except FileExistsError:
            print('Directory already exists.')
            if self.overwrite:
                print('Overwriting directory.')
                shutil.rmtree(self.logs_dir[:-1])
                os.mkdir(self.logs_dir)
            else:
                print('Change to an empty dir or set overwrite == True.')
                raise

        with tf.summary.create_file_writer(self.logs_dir).as_default():
            hp.hparams_config(hparams=self.hparams,
                              metrics=[hp.Metric(self.METRIC_TEST_ACCURACY, display_name='Test Accuracy'),
                                       hp.Metric(self.METRIC_TEST_ACCURACY_UNCERTAINTY, display_name='Test Accuracy Uncertainty'),
                                       hp.Metric(self.METRIC_TRAINING_LOSS, display_name='Training Loss'),
                                       hp.Metric(self.METRIC_VALIDATION_ACCURACY, display_name='Validation Accuracy')])


    def _train_test_model(self,
                          run_dir,
                          Model,
                          udf_hparams,
                          g,
                          features,
                          labels,
                          train_mask,
                          val_mask,
                          test_mask,
                          num_repeats=1):
        '''Runs a training loop given user-defined hyperparameters & performs test.

        Possible user defined hparam keys:

            - HP_NUM_UNITS (name='num_units')
            - HP_NUM_LAYERS (name='num_layers')
            - HP_OPTIMIZER (name='optimizer')
            - HP_LEARNING_RATE (name='learning_rate')
            - HP_NUM_EPOCHS (name='num_epochs')

        '''
        # define default hyperparameters
        HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16]))
        HP_NUM_LAYERS = hp.HParam('num_layers', hp.Discrete([1]))
        HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam']))
        HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.01]))
        HP_NUM_EPOCHS = hp.HParam('num_epochs', hp.Discrete([50]))
        default_hparams = {HP_NUM_UNITS: HP_NUM_UNITS.domain.values[0],
                           HP_NUM_LAYERS: HP_NUM_LAYERS.domain.values[0],        
                           HP_OPTIMIZER: HP_OPTIMIZER.domain.values[0],
                           HP_LEARNING_RATE: HP_LEARNING_RATE.domain.values[0],
                           HP_NUM_EPOCHS: HP_NUM_EPOCHS.domain.values[0],}

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
            if key.name == 'num_units':
                HP_NUM_UNITS = key
            elif key.name == 'num_layers':
                HP_NUM_LAYERS = key
            elif key.name == 'optimizer':
                HP_OPTIMIZER = key
            elif key.name == 'learning_rate':
                HP_LEARNING_RATE = key
            elif key.name == 'num_epochs':
                HP_NUM_EPOCHS = key
            else:
                raise Exception('Unrecognised hyperparameter defined in hparams.')

        # one-hot encode labels
        num_classes = int(len(np.unique(labels)))
        labels = tf.one_hot(indices=labels, depth=num_classes)

        # define gnn model
        layers_config = {'out_feats': [hparams[HP_NUM_UNITS] for _ in range(hparams[HP_NUM_LAYERS])] + [num_classes],
                         'activations': ['relu' for _ in range(hparams[HP_NUM_LAYERS])] + [None]}

        # add edges between each node and itself to preserve old node representations
        g.add_edges(g.nodes(), g.nodes())

        # define optimiser
        if hparams[HP_OPTIMIZER] == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=hparams[HP_LEARNING_RATE])
        else:
            opt = tf.keras.optimizers.SGD(learning_rate=hparams[HP_LEARNING_RATE])

        # init physical device(s)
        device = '/:CPU:0'
        with tf.device(device):

            # init trackers
            all_test_accuracy = []
            all_training_loss = {epoch: [] for epoch in range(hparams[HP_NUM_EPOCHS])}
            all_validation_accuracy = {epoch: [] for epoch in range(hparams[HP_NUM_EPOCHS])}

            for _ in range(num_repeats):
                model = Model(layers_config=layers_config)

                # run training loop
                for epoch in range(hparams[HP_NUM_EPOCHS]):
                    with tf.GradientTape() as tape:
                        logits = model(g, features)
                        loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.boolean_mask(tensor=labels, mask=train_mask),
                                                                       logits=tf.boolean_mask(tensor=logits, mask=train_mask))
                        all_training_loss[epoch].append(tf.math.reduce_mean(loss))
                        grads = tape.gradient(loss, model.trainable_variables)
                        opt.apply_gradients(zip(grads, model.trainable_variables))
                        acc = evaluate(model, g, features, labels, val_mask)
                        all_validation_accuracy[epoch].append(tf.math.reduce_mean(acc))
                # run test
                acc = evaluate(model, g, features, labels, test_mask)
                all_test_accuracy.append(acc)

        # summarise
        # training loss & validation
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

        return mean_accuracy, uncertainty 


    def run(self, run_dir, Model, hparams, data_dict, num_repeats=1):
        '''Trains and tests model with given hparams and tracks with tensorboard.

        Args:
            run_dir (str): Directory to where to save logs + name of this run.
            Model (obj): GNN model to train and test.
            hparams (dict): User-defined hyperparameters to use.
            data_dict (dict): Data for training, validation, and testing.
            num_repeats (int): Number of times to repeat training and testing
                (will return the average accuracy and the uncertainty in the
                accuracy value).

        '''
        data_dict = copy.deepcopy(data_dict) # ensure no overwriting of original data
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams) # record hparams used in this run
            accuracy, uncertainty = self._train_test_model(run_dir, 
                                                           Model,
                                                           hparams, 
                                                           g=data_dict['graph'],
                                                           features=data_dict['features'],
                                                           labels=data_dict['labels'],
                                                           train_mask=data_dict['train_mask'],
                                                           val_mask=data_dict['val_mask'],
                                                           test_mask=data_dict['test_mask'],
                                                           num_repeats=num_repeats)

        with tf.summary.create_file_writer(run_dir).as_default():
            tf.summary.scalar(self.METRIC_TEST_ACCURACY, data=accuracy, step=1)
            tf.summary.scalar(self.METRIC_TEST_ACCURACY_UNCERTAINTY, data=uncertainty, step=1)




