#!/usr/bin/env python

import argparse
import random

import numpy as np
import torch
import torch.optim
import torch.utils.data
from monai.losses import DiceCELoss,DiceLoss
import os

from dataset import DroneImages
from model import UNet_model, SwinUNETR_model
from tqdm import tqdm
from torchmetrics import JaccardIndex
from torch.utils.tensorboard import SummaryWriter

from lr_scheduler import WarmupCosineSchedule

model_zoo = {"unet":UNet_model, "swin":SwinUNETR_model}
loss_zoo = {"dice":DiceLoss, "diceCE":DiceCELoss}
def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def resume_training(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    best_iou = checkpoint["best_val_iou"]
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return best_iou
    

def train(args: argparse.Namespace):
    # set fixed seeds for reproducible execution
    writer = SummaryWriter(args.logdir)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    loss_function = loss_zoo[args.lossfn](to_onehot_y=True, softmax=True)

    # determines the execution device, i.e. CPU or GPU
    device = get_device()
    print(f'Training on {device}')

    # set up the dataset
    drone_images = DroneImages(args.root, in_channels=args.in_channels, return_dict_y=False)
    train_data, test_data = torch.utils.data.random_split(drone_images, [0.8, 0.2])

    # initialize MaskRCNN model
    model = model_zoo[args.model](in_channels=args.in_channels)
    model.to(device)

    # set up optimization procedure
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_iou = 0.
    
    
    # start the actual training procedure
    train_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=args.batch,
                shuffle=True,
                drop_last=False)
    
    warmup_steps = len(train_loader)*5
    t_total = len(train_loader)*args.epochs
    
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)
    
    if args.pretrained is not "none":
        best_iou = resume_training(model, optimizer, scheduler, args.pretrained)

    for epoch in range(args.epochs):
        # set the model into training mode
        model.train()
        
        # training procedure
        train_loss = 0.0
        train_metric = JaccardIndex(task='binary')
        train_metric = train_metric.to(device)

        for i, batch in enumerate(tqdm(train_loader, desc='train')):
            global_step = epoch*len(train_loader) + i
            img, label = batch[0].to(device), batch[1].to(device)
            
            optimizer.zero_grad()
            outputs = model(img)
            loss = loss_function(outputs, label)
            loss.backward()
            optimizer.step()
            scheduler.step()

            
            train_loss += loss.item()

            # compute metric
            with torch.no_grad():
                train_metric(outputs.argmax(axis=1)*1., label.squeeze())
            writer.add_scalar("train/loss", scalar_value=loss.item(), global_step=global_step)
            

        train_loss /= len(train_loader)

        writer.add_scaler("train/IoU", scalar_value=train_metric.compute(), global_step=global_step )
        # set the model in evaluation mode
        model.eval()
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch)

        # test procedure
        test_metric = JaccardIndex(task='binary')
        test_metric = test_metric.to(device)

        test_losses = []
        
        idx_plot = torch.randint(0, len(test_loader))
        for i, batch in enumerate(tqdm(test_loader, desc='test ')):
            img_test, label_test = batch[0].to(device), batch[1].to(device)

            with torch.no_grad():
                test_predictions = model(img_test)
                test_loss = loss_function(test_predictions, label_test)
                test_losses.append(test_loss.item())
                test_metric(test_predictions.argmax(axis=1)*1., label_test.squeeze())
            
            if idx_plot == i:
                for idx_ch, plot_img in enumerate(img_test[0]):
                    writer.add_image(f"valid_img/ch_{idx_ch}", plot_img, global_step, dataformats="HW")
                writer.add_image(f"valid_img/prediction", test_predictions.argmax(axis=1)[0]*1., global_step, dataformats="HW")
                writer.add_image(f"valid_img/ground_truth", label_test.squeeze()[0], global_step, dataformats="HW")
                
        avg_test_loss = torch.mean(test_losses)
        writer.add_scaler("valid/loss", scalar_value=avg_test_loss, global_step=global_step )
        writer.add_scaler("valid/IoU", scalar_value=test_metric.compute(), global_step=global_step )
       
        # output the losses
        print(f'Epoch {epoch}')
        print(f'\tTrain loss: {train_loss}')
        print(f'\tTrain IoU:  {train_metric.compute()}')
        print(f'\tTest IoU:   {test_metric.compute()}')

        # save the best performing model on disk
        if test_metric.compute() > best_iou:
            best_iou = test_metric.compute()
            checkpoint = {}
            checkpoint["best_val_iou"] = best_iou
            checkpoint["best_val_iou_epoch"] = epoch
            checkpoint["model"] = model.state_dict()
            checkpoint["optimizer"] = optimizer.state_dict()
            checkpoint["scheduler"] = scheduler.state_dict()
            
            print('\tSaving better model\n')
            torch.save(checkpoint, args.savename)
        else:
            print('\n')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default=2, help='batch size', type=int)
    parser.add_argument('--in_channels', default=2, help='number of channels', type=int)
    parser.add_argument("--augmentation", action="store_true", help="add this if you want to perform aug")
    parser.add_argument('--epochs', default=100, help='number of training epochs', type=int)
    parser.add_argument('--lr', default=1e-4, help='learning rate of the optimizer', type=float)
    parser.add_argument('--seed', default=42, help='constant random seed for reproduction', type=int)
    parser.add_argument('--outdir', default='checkpoints', help='path to the output root for checkpoint', type=str)
    parser.add_argument('--logdir', default='runs', help='path to the root for logdir', type=str)
    parser.add_argument('--pretrained', default='none', help='path to pretrained checkoint', type=str)
    parser.add_argument('--lossfn', default = "diceCE", choices=['dice', 'diceCE'])
    parser.add_argument('--model', default = "unet", choices=['unet', 'swin'])
    parser.add_argument('--root', default='/hkfs/work/workspace_haic/scratch/qx6387-hida-hackathon-data/train', help='path to the data root', type=str)
    
    
    arguments = parser.parse_args()
    arguments.augmentation = not not arguments.augmentation
    os.makedirs(arguments.outdir, exist_ok=True)
    arguments.savename = arguments.outdir + f"/checkpoint_model_{arguments.model}_loss_{arguments.lossfn}_lr_{arguments.lr}_nepochs_{arguments.epochs}_augment_{arguments.augmentation}_in_channels_{arguments.in_channels}.pt"
    arguments.logdir += f"/logs_model_{arguments.model}_loss_{arguments.lossfn}_lr_{arguments.lr}_nepochs_{arguments.epochs}_augment_{arguments.augmentation}_in_channels_{arguments.in_channels}"
    os.makedirs(arguments.logdir, exist_ok=True)
    
    
    train(arguments)
