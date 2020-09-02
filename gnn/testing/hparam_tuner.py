if __name__ == '__main__':
    import tensorflow as tf
    from tensorboard.plugins.hparams import api as hp
    from gnn.models.tools import load_data
    from gnn.models.models import GCN, GAT
    from gnn.testing.tensorboard_writer import TensorboardWriter
    import time

    # config
    dataset = 'cora'
    logs_dir = '../../data/logs/hparam_tuning/'
    # model = GCN
    model = GAT
    num_repeats = 3 # num times to repeat each trial to get uncertainty value

    # load dataset
    with tf.device('/cpu:0'):
        g, features, labels, train_mask, val_mask, test_mask = load_data(dataset)
        data_dict = {'graph': g,
                     'features': features,
                     'labels': labels,
                     'train_mask': train_mask,
                     'val_mask': val_mask,
                     'test_mask': test_mask}

    # set hyperparams to trial
    HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([64, 128])) # 64
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam'])) # adam
    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.0001])) # 0.0001
    HP_NUM_EPOCHS = hp.HParam('num_epochs', hp.Discrete([300, 400])) # 200
    HP_SHUFFLE = hp.HParam('shuffle', hp.Discrete([True])) 
    HP_BATCH_NORM = hp.HParam('batch_norm', hp.Discrete([False]))
    HP_DROPOUT_RATE = hp.HParam('dropout_rate', hp.Discrete([0])) 
    HP_SAMPLE = hp.HParam('sample', hp.Discrete([True])) # True
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([35])) 
    HP_NUM_NEIGHBOURS = hp.HParam('num_neighbours', hp.Discrete([4])) 
    HP_NUM_HEADS = hp.HParam('num_heads', hp.Discrete([3, 4, 5, 8]))
    hparams = [HP_NUM_UNITS, 
               HP_OPTIMIZER, 
               HP_LEARNING_RATE, 
               HP_NUM_EPOCHS, 
               HP_SHUFFLE,
               HP_BATCH_NORM,
               HP_DROPOUT_RATE,
               HP_SAMPLE,
               HP_BATCH_SIZE,
               HP_NUM_NEIGHBOURS,
               HP_NUM_HEADS]
    num_runs = 1
    for param in hparams:
        num_runs *= len(param.domain.values)

    # init tensorboard
    tboard = TensorboardWriter(logs_dir, hparams, overwrite=True)

    # trial each combination of hyperparams
    session_num = 1
    started = time.time()
    for num_units in HP_NUM_UNITS.domain.values:
        for optimizer in HP_OPTIMIZER.domain.values:
            for learning_rate in HP_LEARNING_RATE.domain.values:
                for num_epochs in HP_NUM_EPOCHS.domain.values:
                    for shuffle in HP_SHUFFLE.domain.values:
                        for batch_norm in HP_BATCH_NORM.domain.values:
                            for dropout_rate in HP_DROPOUT_RATE.domain.values:
                                for sample in HP_SAMPLE.domain.values:
                                    for batch_size in HP_BATCH_SIZE.domain.values:
                                        for num_neighbours in HP_NUM_NEIGHBOURS.domain.values:
                                            for num_heads in HP_NUM_HEADS.domain.values:
                                                hparams = {HP_NUM_UNITS: num_units,
                                                           HP_OPTIMIZER: optimizer,
                                                           HP_LEARNING_RATE: learning_rate,
                                                           HP_NUM_EPOCHS: num_epochs,
                                                           HP_SHUFFLE: shuffle,
                                                           HP_BATCH_NORM: batch_norm,
                                                           HP_DROPOUT_RATE: dropout_rate,
                                                           HP_SAMPLE: sample,
                                                           HP_BATCH_SIZE: batch_size,
                                                           HP_NUM_NEIGHBOURS: num_neighbours,
                                                           HP_NUM_HEADS: num_heads,}
                                                run_name = 'run_'+str(session_num)
                                                print('Starting trial {} of {} ({} repeats)'.format(run_name, num_runs, num_repeats))
                                                start = time.time()
                                                print({h.name: hparams[h] for h in hparams})
                                                tboard.run(logs_dir + run_name, model, hparams, data_dict, num_repeats=num_repeats)
                                                end = time.time()
                                                print('Finished trial in {} s'.format(end-start))
                                                session_num += 1
    ended = time.time()
    print('Finished hparam trials in {} s'.format(ended-started))





