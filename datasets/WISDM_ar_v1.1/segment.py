import os
import pickle as pkl

import numpy as np
import pandas as pd

np.random.seed(17)

# there are 6 empty lines in WISDM_ar_v1.1_raw.txt
df = pd.read_csv(
    './WISDM_ar_v1.1_raw.txt',
    header=None,
    names=['uid', 'activity', 'timestamp', 'x', 'y', 'z'],
    # skip_blank_lines=False,
)

df['z'] = df['z'].str.replace(';', '').astype('float64')

df = df.interpolate()

df['x_normed'] = (df.x - df.x.min()) / (df.x.max() - df.x.min())
df['y_normed'] = (df.y - df.y.min()) / (df.y.max() - df.y.min())
df['z_normed'] = (df.z - df.z.min()) / (df.z.max() - df.z.min())

activity_to_label = dict(zip(df.activity.unique(), range(df.activity.nunique())))
print(activity_to_label)
df['label'] = df.activity.map(activity_to_label)

df_train = df[df.uid <= 30].reset_index(drop=True)
df_test = df[df.uid > 30].reset_index(drop=True)


def segment(df, L=128, stride=64, dt_tol=1e3):
    kinks = np.unique(
        np.concatenate(
            (
                [0],
                np.where(df.uid[1:].values != df.uid[:-1].values)[0] + 1,
                np.where(df.label[1:].values != df.label[:-1].values)[0] + 1,
                [len(df) - 1],
            )
        )
    )
    kinks.sort()

    segments = []
    fragments = []

    for i, j in zip(kinks[:-1], kinks[1:]):
        for k in range(i, j, stride):
            if k + L < j:
                segments.append(df[k : k + L][['x_normed', 'y_normed', 'z_normed', 'label']].values.tolist())

            else:
                fragments.append(df[k:j][['x_normed', 'y_normed', 'z_normed', 'label']].values.tolist())
                break

    return segments, fragments


if not os.path.exists('train.pkl'):
    segments_train, fragments_train = segment(df_train)
    with open('train.pkl', 'wb') as f:
        pkl.dump({'segments': segments_train, 'fragments': fragments_train}, f)

if not os.path.exists('test.pkl'):
    segments_test, fragments_test = segment(df_test)
    with open('test.pkl', 'wb') as f:
        pkl.dump({'segments': segments_test, 'fragments': fragments_test}, f)
