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

from model import UNet_model, SwinUNETR_model, EfficientUNet_model, UNet_small_model, EfficientUNet_small_model

model_zoo = {"unet":UNet_model, "swin":SwinUNETR_model, "effunet": EfficientUNet_model,
             "unet_small":UNet_small_model, "effunet_small":EfficientUNet_small_model}


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(args: argparse.Namespace):
    # set fixed seeds for reproducible execution
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # determines the execution device, i.e. CPU or GPU
    device = get_device()
    print(f'Training on {device}')

    # set up the dataset
    drone_images = get_DroneImages_datalist(args.root, predict=True)
    test_data = get_DroneImages_dataset(drone_images, augmentation = False, in_channels = args.in_channels)

    # initialize the U-Net model
    model = model_zoo[args.model](in_channels=args.in_channels)
    model.to(device)
    
    if args.checkpoint:
        print(f'Restoring model checkpoint from {args.checkpoint}')
        model.load_state_dict(torch.load(args.checkpoint)["model"])
    model.to(device)

    # set the model in evaluation mode
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch)

    # test procedure
    test_metric = JaccardIndex(task='binary')
    test_metric = test_metric.to(device)

    for i, batch in enumerate(tqdm(test_loader, desc='test ')):
        img_test, label_test = batch["image"].to(device), batch["label"].to(device)

        # score_threshold = 0.7
        with torch.no_grad():
            test_predictions = model(img_test)
            test_metric(test_predictions.argmax(axis=1)*1., label_test[:, 0, :, :])

    print(f'Test IoU: {test_metric.compute()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch', default=1, help='batch size', type=int)
    parser.add_argument('--in_channels', default=2, help='number of channels', type=int)
    parser.add_argument('--model', default = "unet", choices=['unet', 'swin', 'effunet', 'unet_small', 'effunet_small'])
    parser.add_argument('-c', '--checkpoint', default='checkpoint.pt', help='model checkpoint', type=str)
    parser.add_argument('-s', '--seed', default=42, help='constant random seed for reproduction', type=int)
    parser.add_argument('root', help='path to the data root', type=str)

    arguments = parser.parse_args()
    predict(arguments)
