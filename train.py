import argparse
import pickle as pkl
import time

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(17)


with open('./datasets/WISDM_ar_v1.1/train.pkl', 'rb') as f:
    segments_train, fragments_train = pkl.load(f).values()

with open('./datasets/WISDM_ar_v1.1/test.pkl', 'rb') as f:
    segments_test, fragments_test = pkl.load(f).values()


class HARDataset(Dataset):
    def __init__(self, segments):
        self.segments = segments

    def __getitem__(self, index):
        segment = self.segments[index]
        return torch.Tensor(segment)[:, :-1], int(segment[0][-1])

    def __len__(self):
        return len(self.segments)


# can achieve 92% accuracy
class LSTMCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.lstm = nn.LSTM(3, 32, 1)

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, (1, 5), stride=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
            nn.Conv2d(64, 128, (1, 3), stride=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # N x L x 3
        out, _ = self.lstm(x.transpose(0, 1))
        # L x N x 32
        out = out.transpose(0, 1).unsqueeze(1)
        # N x 1 x L x 32
        out = self.cnn(out).view(out.size(0), -1)
        # N x 128
        out = self.classifier(out)
        # N x num_classes
        return out


# light & fast, can achieve 85%
class MLP(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(384, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.feature(x.view(x.size(0), -1))
        out = self.classifier(out)
        return out


def run_epoch(loader, model, optimizer, criterion, epoch, gpu_mode):
    model.train()

    for i_batch, data_batch in enumerate(loader):
        inputs, labels = data_batch

        if gpu_mode:
            inputs = inputs.cuda()
            labels = labels.cuda()

        t0 = time.time()

        # forward
        optimizer.zero_grad()
        emb = model(inputs)
        loss = criterion(emb, labels)

        t1 = time.time()
        t_fwd = t1 - t0
        t0 = t1

        # backward
        loss.backward()

        optimizer.step()

        t1 = time.time()
        t_bwd = t1 - t0

        print(f'epoch: {epoch}, iter: {i_batch}, loss: {loss:10.8f}')


def test(loader, model, gpu_mode):
    model.eval()

    conf = None
    gt_pred_scr = []
    ith = 0

    for i_batch, data_batch in enumerate(loader):
        inputs, labels = data_batch

        if gpu_mode:
            inputs = inputs.cuda()
            labels = labels.cuda()

        with torch.no_grad():
            scr = model(inputs)

            if conf is None:
                sz = scr.size(1)
                conf = torch.zeros(sz, sz)

            _, pred = torch.max(scr, 1)
            for i, j in zip(labels.data, pred.data):
                conf[i, j] += 1

            gt_pred_scr.append(
                torch.cat([labels.unsqueeze(1).type_as(scr), pred.unsqueeze(1).type_as(scr), scr], dim=1)
            )

        ith += len(labels)
        # if not ith % 1000:
        #     print(f'progressed to test sample {ith}')

    # print(f'progressed to test sample {ith}')
    gt_pred_scr = torch.cat(gt_pred_scr)

    recall = conf.diag() / (conf.sum(1) + 1e-12)
    precision = conf.diag() / (conf.sum(0) + 1e-12)
    acc = conf.diag().sum() / (conf.sum() + 1e-12)

    big_conf = torch.zeros(conf.size(0) + 1, conf.size(1) + 1)
    big_conf[: conf.size(0), : conf.size(1)] = conf
    big_conf[: conf.size(0), -1] = recall
    big_conf[-1, : conf.size(1)] = precision
    big_conf[-1, -1] = acc

    # print(f'top1 acc: {acc:.3f}, conf: \n{big_conf}')

    return recall, precision, acc, gt_pred_scr


def train(
    n_epochs=30,
    test_epochs=1,
    save_epochs=5,
    reduce_epochs=[15, 25],
    batch_size=100,
    num_classes=6,
    init_lr=1e-3,
    resume=None,
    mode='TRAIN',
    save_prefix='run',
    gpu_mode=True,
):
    train_dataset = HARDataset(segments_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dataset = HARDataset(segments_test)
    test_dataloader = DataLoader(test_dataset, batch_size=2 * batch_size, shuffle=False, drop_last=False)

    model = LSTMCNN(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()

    if gpu_mode:
        model = model.cuda()
        criterion = criterion.cuda()

    optimizer = optim.Adam(model.parameters(), lr=init_lr)

    start_epoch = 0
    if resume:
        checkpoint = torch.load(resume, map_location='cpu')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint

    print(f'epoch {start_epoch} initial learning rate')
    for param_group in optimizer.param_groups:
        print(f'param group: {param_group["lr"]:10.8f}')

    for epoch in range(start_epoch, n_epochs + 1):
        if (epoch > start_epoch and not epoch % test_epochs) or (
            epoch == start_epoch and mode in ['TEST', 'TESTTRAIN']
        ):
            # test
            recall, precision, acc, gt_pred_scr = test(test_dataloader, model, gpu_mode)
            print(
                f'epoch {epoch}, accuracy {acc:10.8f}, recall-precision:'
                f' \n{torch.cat([recall.view(1, -1), precision.view(1, -1)])}'
            )

            if mode == 'TEST':
                return

        if not epoch % save_epochs and epoch > start_epoch:
            # save
            torch.save(
                {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                f'cache/models/{save_prefix}_epoch_{epoch}.state',
            )

        # decrease lr if specified
        if epoch in reduce_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.2

            print(f'epoch {start_epoch} decrease learning rate')
            for param_group in optimizer.param_groups:
                print(f'param group: {param_group["lr"]:10.8f}')

        run_epoch(
            loader=train_dataloader,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
            gpu_mode=gpu_mode,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trainer')

    # settings
    parser.add_argument('--n_epochs', type=int, help='number of epochs in total to train.')
    parser.add_argument(
        '--reduce_epochs',
        type=int,
        nargs='+',
        help='epochs when learning rate is multiplier by 0.2.',
    )
    parser.add_argument('--test_epochs', type=int, help='interval of epochs to do testing.')
    parser.add_argument('--save_epochs', type=int, help='interval of epochs to save checkpoints.')
    parser.add_argument('--save_prefix', type=str, help='prefix of saved checkpoints.')

    parser.add_argument('--num_classes', type=int, help='number of classes.')
    parser.add_argument('--batch_size', type=int, help='batch size.')
    parser.add_argument('--init_lr', type=float, help='initial learning rate.')

    parser.add_argument('--resume', type=str, help='checkpoint to resume training (or testing).')
    parser.add_argument(
        '--mode',
        default='TESTTRAIN',
        type=str.upper,
        choices=['TEST', 'TESTTRAIN', 'TRAIN'],
        help=(
            'mode options: '
            'TEST: only do testing; '
            'TESTTRAIN: do testing first and then start/continue training; '
            'TRAIN: without testing first, directly start/continue training. default "TESTTRAIN"'
        ),
    )

    args = vars(parser.parse_args())

    config = {}
    with open('./config.yaml', 'rb') as f:
        config = yaml.safe_load(f)

    config.update({k: v for k, v in args.items() if v is not None})

    print(config)
    train(**config)
