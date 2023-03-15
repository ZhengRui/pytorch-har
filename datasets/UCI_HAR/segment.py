import os
import pickle as pkl

import numpy as np
import pandas as pd


def segment(split='train'):
    labels = pd.read_csv(f'./{split}/y_{split}.txt', header=None).to_numpy() - 1

    SIGNALS = [
        'body_acc_x',
        'body_acc_y',
        'body_acc_z',
        'body_gyro_x',
        'body_gyro_y',
        'body_gyro_z',
        'total_acc_x',
        'total_acc_y',
        'total_acc_z',
    ]

    signals = []

    for signal in SIGNALS:
        filename = f'{split}/Inertial Signals/{signal}_{split}.txt'
        signals.append(pd.read_csv(filename, header=None, delim_whitespace=True).to_numpy())

    signals = np.transpose(signals, (1, 2, 0))

    segments = np.concatenate((signals, np.expand_dims(labels.repeat(signals.shape[1], axis=1), axis=2)), axis=2)

    return segments


segments_train = segment('train')
segments_test = segment('test')

# if do 0-1 normalization
# segments = np.concatenate((segments_train, segments_test), axis=0).reshape(-1, segments_train.shape[-1])
# signals_max = segments[:, :-1].max(axis=0, keepdims=True)
# signals_min = segments[:, :-1].min(axis=0, keepdims=True)
# segments[:, :-1] = (segments[:, :-1] - signals_min) / (signals_max - signals_min)
# segments = segments.reshape(-1, *segments_train.shape[1:])
# segments_train = segments[: len(segments_train)]
# segments_test = segments[len(segments_train) :]


if not os.path.exists('train.pkl'):
    with open('train.pkl', 'wb') as f:
        pkl.dump({'segments': segments_train, 'fragments': None}, f)

if not os.path.exists('test.pkl'):
    with open('test.pkl', 'wb') as f:
        pkl.dump({'segments': segments_test, 'fragments': None}, f)
