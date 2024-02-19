#!/usr/bin/env python

import argparse
import random

import numpy as np
import torch
import torch.utils.data

from dataset import DroneImages
from torchmetrics import JaccardIndex
from model import MaskRCNN
from tqdm import tqdm


def collate_fn(batch) -> tuple:
    return tuple(zip(*batch))


def instance_to_semantic_mask(pred, target):
    pred_mask = torch.stack([p['masks'].sum(dim=0).clamp(0., 1.).squeeze() for p in pred])  # [batch_size, width, height]
    target_mask = torch.stack([t['masks'].sum(dim=0).clamp(0., 1.).squeeze() for t in target])  # [batch_size, width, height]

    return pred_mask, target_mask


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(hyperparameters: argparse.Namespace):
    # set fixed seeds for reproducible execution
    random.seed(hyperparameters.seed)
    np.random.seed(hyperparameters.seed)
    torch.manual_seed(hyperparameters.seed)

    # determines the execution device, i.e. CPU or GPU
    device = get_device()
    print(f'Training on {device}')

    # set up the dataset
    drone_images = DroneImages(hyperparameters.root)
    test_data = drone_images

    # initialize the U-Net model
    model = MaskRCNN()
    if hyperparameters.model:
        print(f'Restoring model checkpoint from {hyperparameters.model}')
        model.load_state_dict(torch.load(hyperparameters.model))
    model.to(device)

    # set the model in evaluation mode
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=hyperparameters.batch, collate_fn=collate_fn)

    # test procedure
    test_metric = JaccardIndex(task='binary')
    test_metric = test_metric.to(device)

    for i, batch in enumerate(tqdm(test_loader, desc='test ')):
        x_test, test_label = batch
        x_test = list(image.to(device) for image in x_test)
        test_label = [{k: v.to(device) for k, v in l.items()} for l in test_label]

        # score_threshold = 0.7
        with torch.no_grad():
            test_predictions = model(x_test)

            test_metric(*instance_to_semantic_mask(test_predictions, test_label))

    print(f'Test IoU: {test_metric.compute()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch', default=1, help='batch size', type=int)
    parser.add_argument('-m', '--model', default='checkpoint.pt', help='model checkpoint', type=str)
    parser.add_argument('-s', '--seed', default=42, help='constant random seed for reproduction', type=int)
    parser.add_argument('root', help='path to the data root', type=str)

    arguments = parser.parse_args()
    predict(arguments)
