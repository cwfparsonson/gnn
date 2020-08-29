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
    model = GCN
    num_repeats = 3 

    # load dataset
    g, features, labels, train_mask, val_mask, test_mask = load_data(dataset)
    data_dict = {'graph': g,
                 'features': features,
                 'labels': labels,
                 'train_mask': train_mask,
                 'val_mask': val_mask,
                 'test_mask': test_mask}

    # set hyperparams to trial
    HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32])) #32
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam']))
    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.001, 0.0001])) #0.001
    HP_NUM_EPOCHS = hp.HParam('num_epochs', hp.Discrete([300, 600])) #300
    hparams = [HP_NUM_UNITS, HP_OPTIMIZER, HP_LEARNING_RATE, HP_NUM_EPOCHS]

    # init tensorboard
    tboard = TensorboardWriter(logs_dir, hparams, overwrite=True)

    # trial each combination of hyperparams
    session_num = 0
    started = time.time()
    for num_units in HP_NUM_UNITS.domain.values:
        for optimizer in HP_OPTIMIZER.domain.values:
            for learning_rate in HP_LEARNING_RATE.domain.values:
                for num_epochs in HP_NUM_EPOCHS.domain.values:
                    hparams = {HP_NUM_UNITS: num_units,
                               HP_OPTIMIZER: optimizer,
                               HP_LEARNING_RATE: learning_rate,
                               HP_NUM_EPOCHS: num_epochs,}
                    run_name = 'run_'+str(session_num)
                    print('Starting trial: {} ({} repeats)'.format(run_name, num_repeats))
                    start = time.time()
                    print({h.name: hparams[h] for h in hparams})
                    tboard.run(logs_dir + run_name, model, hparams, data_dict, num_repeats=num_repeats)
                    end = time.time()
                    print('Finished trial in {} s'.format(end-start))
                    session_num += 1
    ended = time.time()
    print('Finished hparam trials in {} s'.format(ended-started))





