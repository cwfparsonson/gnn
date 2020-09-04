if __name__ == '__main__':
    import tensorflow as tf
    from tensorboard.plugins.hparams import api as hp
    import numpy as np
    from gnn.models.models import GCN, GAT
    from gnn.testing.tensorboard_writer import TensorboardWriter
    import time
    import json


    # _________________________________________________________________________
    # BASIC CONFIGURATION
    # _________________________________________________________________________
    save_dir = '../../data/logs/dataset_loop_test/'
    num_repeats = 0 # num times to repeat each trial to get uncertainty value
    save_model = True # whether or not to save trained model after each trial
    overwrite = True # whether or not to overwrite prev saved data

    # _________________________________________________________________________
    # HYPERPARAMETERS TO TRIAL
    # _________________________________________________________________________
    # -------------------------------------------------------------------------
    # gnn layer configuration (specific to GNN), uncomment one below
    # -------------------------------------------------------------------------

    # GCN
    model = GCN
    out_feats = [[64, 'auto']]
    activations = activations = [['leaky_relu', None]]
    dropout_rates = [[None, None]]
    batch_norms = [[False, False]]
    layers_configs = []
    for o_fs in out_feats:
        for acts in activations:
            for drs in dropout_rates:
                for bns in batch_norms:
                    layers_config = {'out_feats': o_fs,
                                     'activations': acts,
                                     'dropout_rates': drs,
                                     'batch_norms': bns}
                    json_layers_config = json.dumps(layers_config) # conv to str
                    layers_configs.append(json_layers_config)

    # # GAT
    # model = GAT
    # out_feats = [[64, 7]]
    # num_heads = [[8, 1], [10, 1]]
    # layers_configs = []
    # for o_fs in out_feats:
        # for heads in num_heads:
            # layers_config = {'out_feats': o_fs,
                             # 'num_heads': heads}
            # json_layers_config = json.dumps(layers_config) # conv to str
            # layers_configs.append(json_layers_config)

    # -------------------------------------------------------------------------
    # general configuration
    # -------------------------------------------------------------------------
    HP_LAYERS_CONFIG = hp.HParam('layers_config', hp.Discrete(layers_configs))
    HP_DATASET = hp.HParam('dataset', hp.Discrete(['cora', 'pubmed']))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam'])) # adam
    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.0001])) # 0.0001
    HP_NUM_EPOCHS = hp.HParam('num_epochs', hp.Discrete([200])) # 200 # 300
    HP_SHUFFLE = hp.HParam('shuffle', hp.Discrete([True])) 
    HP_SAMPLE = hp.HParam('sample', hp.Discrete([True])) # True
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([35])) # 35
    HP_NUM_NEIGHBOURS = hp.HParam('num_neighbours', hp.Discrete([4]))  # 4
    hparams = [HP_LAYERS_CONFIG,
               HP_DATASET,
               HP_OPTIMIZER, 
               HP_LEARNING_RATE, 
               HP_NUM_EPOCHS, 
               HP_SHUFFLE,
               HP_SAMPLE,
               HP_BATCH_SIZE,
               HP_NUM_NEIGHBOURS]


    # _________________________________________________________________________
    # BACKEND
    # _________________________________________________________________________
    tboard = TensorboardWriter(save_dir, hparams, save_model=save_model, overwrite=overwrite)

    # trial each combination of hyperparams
    num_runs = 1
    for param in hparams:
        num_runs *= len(param.domain.values)
    session_num = 1
    started = time.time()
    for layers_config in HP_LAYERS_CONFIG.domain.values:
        for dataset in HP_DATASET.domain.values:
            for optimizer in HP_OPTIMIZER.domain.values:
                for learning_rate in HP_LEARNING_RATE.domain.values:
                    for num_epochs in HP_NUM_EPOCHS.domain.values:
                        for shuffle in HP_SHUFFLE.domain.values:
                            for sample in HP_SAMPLE.domain.values:
                                for batch_size in HP_BATCH_SIZE.domain.values:
                                    for num_neighbours in HP_NUM_NEIGHBOURS.domain.values:
                                        hparams = {HP_LAYERS_CONFIG: layers_config,
                                                   HP_DATASET: dataset,
                                                   HP_OPTIMIZER: optimizer,
                                                   HP_LEARNING_RATE: learning_rate,
                                                   HP_NUM_EPOCHS: num_epochs,
                                                   HP_SHUFFLE: shuffle,
                                                   HP_SAMPLE: sample,
                                                   HP_BATCH_SIZE: batch_size,
                                                   HP_NUM_NEIGHBOURS: num_neighbours,}
                                        run_name = 'run_'+str(session_num)
                                        print('Starting trial {} of {} ({} repeats)'.format(run_name, num_runs, num_repeats))
                                        start = time.time()
                                        print({h.name: hparams[h] for h in hparams})
                                        tboard.run(run_name, model, hparams, num_repeats=num_repeats)
                                        end = time.time()
                                        print('Finished trial in {} s'.format(end-start))
                                        session_num += 1
    ended = time.time()
    print('Finished hparam trials in {} s'.format(ended-started))





