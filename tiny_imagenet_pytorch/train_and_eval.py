import os
import numpy as np
import torch.nn as nn
import torch
import argparse
import time
from torch.autograd import Variable
from tensorboard_logger import configure, log_value
from clusterone import get_logs_path, get_data_path
from .model.model import fetch_metrics, TinyImageNetModel
from .model.data_loader import fetch_label_map, fetch_dataloader

TRAIN_DATA_DIR = get_data_path(
    dataset_name = 'artem-towa/artem-tiny-imagenet-example',
    local_root = os.path.expanduser('~/Documents/Scratch/tiny_imagenet/'),
    local_repo = 'tiny-imagenet-200',
    path = 'train'
)
EVAL_DATA_DIR = get_data_path(
    dataset_name = 'artem-towa/artem-tiny-imagenet-example',
    local_root = os.path.expanduser('~/Documents/Scratch/tiny_imagenet/'),
    local_repo = 'tiny-imagenet-200',
    path = 'val/for_keras'
)
UNIQUE_LABELS_PATH = get_data_path(
    dataset_name = 'artem-towa/artem-tiny-imagenet-example',
    local_root = os.path.expanduser('~/Documents/Scratch/tiny_imagenet/'),
    local_repo = 'tiny-imagenet-200',
    path = 'wnids.txt'
)
LOGS_PATH = get_logs_path('./logs')

configure(LOGS_PATH)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs', default=1, type=int)
parser.add_argument('--save_summary_steps', default=50, type=int)
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
if args.cuda:
    device = torch.device('cuda:0')
    print('Using GPU')
else:
    device = torch.device('cpu')
    print('CUDA not found')

# Set the random seed for reproducible experiments
torch.manual_seed(230)
if args.cuda: torch.cuda.manual_seed(230)

def train(model, dataloader, loss_fn, metrics, optimizer, save_summary_steps=50):
    model.train()
    summ = []
    for i, (x_batch, labels_batch) in enumerate(dataloader):
        if args.cuda:
            x_batch = x_batch.to(device)
            labels_batch = labels_batch.to(device)

        x_batch = Variable(x_batch)
        labels_batch = Variable(labels_batch)

        output_batch = model(x_batch)

        loss = loss_fn(output_batch, labels_batch)
        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()

        if i % save_summary_steps == 0:
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            summ_batch = {}
            summ_batch['loss'] = loss.data.item()
            for metric in metrics:
                summ_batch[metric] = metrics[metric](labels_batch, output_batch)

            summ.append(summ_batch)

    metric_means = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metric_str = "TRAIN -- "
    for metric in metric_means:
        metric_str += "{0}: {1} ".format(metric, metric_means[metric])

    print(metric_str)
    return metric_means

def eval(model, dataloader, loss_fn, metrics):
    model.eval()
    summ = []
    for i, (x_batch, labels_batch) in enumerate(dataloader):
        if args.cuda:
            x_batch = x_batch.cuda(async=True)
            labels_batch = labels_batch.cuda(async=True)

        x_batch = Variable(x_batch)
        labels_batch = Variable(labels_batch)

        output_batch = model(x_batch)
        loss = loss_fn(output_batch, labels_batch)

        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()
        summ_batch = {}
        summ_batch['loss'] = loss.data.item()
        for metric in metrics:
            summ_batch[metric] = metrics[metric](labels_batch, output_batch)

        summ.append(summ_batch)

    metric_means = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metric_str = "EVAL -- "
    for metric in metric_means:
        metric_str += "{0}: {1} ".format(metric, metric_means[metric])

    print(metric_str)
    return metric_means

def train_and_eval():
    label_map = fetch_label_map(UNIQUE_LABELS_PATH)
    train_dataloader = fetch_dataloader(
        'train', TRAIN_DATA_DIR, label_map,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.cuda
    )
    eval_dataloader = fetch_dataloader(
        'eval', EVAL_DATA_DIR, label_map,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.cuda
    )
    metrics = fetch_metrics()

    model = TinyImageNetModel().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for i in range(args.num_epochs):
        epoch_start_time = time.time()
        metric_means = train(model, train_dataloader, loss_fn, metrics, optimizer, save_summary_steps=args.save_summary_steps)
        for tag, val in metric_means.items():
            log_value('training ' + tag, val, i + 1)

        print('Finished training epoch {0} in {1} min'.format(i, str((time.time() - epoch_start_time) / 60.)))

        metric_means = eval(model, eval_dataloader, loss_fn, metrics)
        for tag, val in metric_means.items():
            log_value('eval ' + tag, val, i + 1)

if __name__ == '__main__':
    train_and_eval()