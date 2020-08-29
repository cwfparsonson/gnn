if __name__ == '__main__':
    import tensorflow as tf
    from tensorboard.plugins.hparams import api as hp
    from gnn.models.tools import load_data
    from gnn.models.graph_conv import GCN
    from gnn.testing.tensorboard_writer import TensorboardWriter
    import time

    dataset = 'cora'
    logs_dir = '../../data/logs/hparam_tuning/'

    # load dataset
    g, features, labels, train_mask, val_mask, test_mask = load_data(dataset)
    data_dict = {'graph': g,
                 'features': features,
                 'labels': labels,
                 'train_mask': train_mask,
                 'val_mask': val_mask,
                 'test_mask': test_mask}

    # hyperparams
    HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([4, 8, 16, 32, 64]))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['sgd', 'adam']))
    HP_NUM_EPOCHS = hp.HParam('num_epochs', hp.Discrete([50, 100, 150]))
    hparams = [HP_NUM_UNITS, HP_OPTIMIZER, HP_NUM_EPOCHS]

    # init tensorboard
    tboard = TensorboardWriter(logs_dir, hparams, overwrite=True)

    session_num = 0
    started = time.time()
    for num_units in HP_NUM_UNITS.domain.values:
        for optimizer in HP_OPTIMIZER.domain.values:
            for num_epochs in HP_NUM_EPOCHS.domain.values:
                hparams = {HP_NUM_UNITS: num_units,
                           HP_OPTIMIZER: optimizer,
                           HP_NUM_EPOCHS: num_epochs,}
                run_name = 'run_'+str(session_num)
                print('Starting trial: {}'.format(run_name))
                start = time.time()
                print({h.name: hparams[h] for h in hparams})
                tboard.run(logs_dir + run_name, GCN, hparams, data_dict)
                end = time.time()
                print('Finished trial in {} s'.format(end-start))
                session_num += 1
    ended = time.time()
    print('Finished hparam trials in {} s'.format(ended-started))





