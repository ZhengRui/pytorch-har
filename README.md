This repo is to replicate results from paper: LSTM-CNN Architecture for Human Activity Recognition

#### Datasets folder structure
```bash
datasets
├── UCI_HAR
│   ├── README.txt
│   ├── activity_labels.txt
│   ├── features.txt
│   ├── features_info.txt
│   ├── segment.py
│   ├── test
│   │   ├── Inertial Signals
│   │   │   ├── body_acc_x_test.txt
│   │   │   ├── body_acc_y_test.txt
│   │   │   ├── body_acc_z_test.txt
│   │   │   ├── body_gyro_x_test.txt
│   │   │   ├── body_gyro_y_test.txt
│   │   │   ├── body_gyro_z_test.txt
│   │   │   ├── total_acc_x_test.txt
│   │   │   ├── total_acc_y_test.txt
│   │   │   └── total_acc_z_test.txt
│   │   ├── X_test.txt
│   │   ├── subject_test.txt
│   │   └── y_test.txt
│   ├── test.pkl
│   ├── test_raw.pkl
│   ├── train
│   │   ├── Inertial Signals
│   │   │   ├── body_acc_x_train.txt
│   │   │   ├── body_acc_y_train.txt
│   │   │   ├── body_acc_z_train.txt
│   │   │   ├── body_gyro_x_train.txt
│   │   │   ├── body_gyro_y_train.txt
│   │   │   ├── body_gyro_z_train.txt
│   │   │   ├── total_acc_x_train.txt
│   │   │   ├── total_acc_y_train.txt
│   │   │   └── total_acc_z_train.txt
│   │   ├── X_train.txt
│   │   ├── subject_train.txt
│   │   └── y_train.txt
│   ├── train.pkl
│   └── train_raw.pkl
└── WISDM_ar_v1.1
    ├── WISDM_ar_v1.1_raw.txt
    ├── WISDM_ar_v1.1_raw_about.txt
    ├── WISDM_ar_v1.1_trans_about.txt
    ├── WISDM_ar_v1.1_transformed.arff
    ├── readme.txt
    ├── segment.py
    ├── test.pkl
    └── train.pkl
```

#### For `WISDM_ar_v1.1` dataset

- generate segments for train/test split: the generated segments are normalized to `0-1`.
    ```bash
    cd datasets/WISDM_ar_v1.1
    python segment.py
    ```

- train: can achieve `95%-96%` top-1 accuracy, similar to paper result.
    ```bash
    python train.py --batch_size 1000 --mode TESTTRAIN --init_lr .01
    ```

#### For `UCI_HAR` dataset

- generate segments for train/test split: the generated segments are not normalized, if you want normalized signals, uncomment relevant lines in `segment.py`.
    ```bash
    cd datasets/UCI_HAR
    python segment.py
    ```

- train: can achieve `92%-93%` top-1 accuracy, worse than paper result.
    ```bash
    python train.py --batch_size 1000 --mode TESTTRAIN --init_lr .01
    ```

#### Gotchas
- batch norm layer is important
- gradients passed to lstm layers are very small, a 2-layer lstm is harder to learn than 1-layer lstm, and indeed 1-layer lstm achieves better result
- Adam optimizer is better than SGD

Tried to just use MLP, it can achieve `85%` top-1 accuracy on `WISDM_ar_v1.1`.
