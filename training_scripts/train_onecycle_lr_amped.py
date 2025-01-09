import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
import time
from math import sqrt

from model import ResNet50
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Params:
    def __init__(self):
        self.batch_size = 256
        self.name = "resnet_50_onecycle"
        self.workers = 12
        self.max_lr = 0.11
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.epochs = 100
        self.pct_start = 0.3
        self.div_factor = 25.0
        self.final_div_factor = 1e4

    def __repr__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

class MetricLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics = []
        
    def log_metrics(self, epoch_metrics):
        self.metrics.append(epoch_metrics)
        with open(os.path.join(self.log_dir, 'training_log.json'), 'w') as f:
            json.dump(self.metrics, f, indent=4)

def train(dataloader, model, loss_fn, optimizer, epoch, writer, scaler, metric_logger):
    size = len(dataloader.dataset)
    model.train()
    start0 = time.time()
    running_loss = 0.0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")
    
    for batch, (X, y) in progress_bar:
        X, y = X.to(device), y.to(device)
        
        with autocast():
            pred = model(X)
            loss = loss_fn(pred, y)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        running_loss += loss.item()
        batch_size = len(X)
        step = epoch * size + (batch + 1) * batch_size

        if batch % 100 == 0:
            current_loss = loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                "loss": current_loss, 
                "lr": f"{current_lr:.6f}",
                "progress": f"{(batch+1)*batch_size}/{size}"
            })
            if writer is not None:
                writer.add_scalar('training loss', current_loss, step)
                writer.add_scalar('learning rate', current_lr, step)

    epoch_time = time.time() - start0
    avg_loss = running_loss / len(dataloader)
    
    metrics = {
        'epoch': epoch + 1,
        'train_loss': avg_loss,
        'learning_rate': optimizer.param_groups[0]['lr'],
        'epoch_time': epoch_time
    }
    metric_logger.log_metrics(metrics)
    
    logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")

def test(dataloader, model, loss_fn, epoch, writer, train_dataloader, metric_logger, calc_acc5=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct, correct_top5 = 0, 0, 0

    progress_bar = tqdm(dataloader, desc=f"Testing Epoch {epoch+1}")

    with torch.no_grad():
        with autocast():
            for X, y in progress_bar:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

                if calc_acc5:
                    _, pred_top5 = pred.topk(5, 1, largest=True, sorted=True)
                    correct_top5 += pred_top5.eq(y.view(-1, 1).expand_as(pred_top5)).sum().item()

    test_loss /= num_batches
    accuracy = 100 * correct / size
    top5_accuracy = 100 * correct_top5 / size if calc_acc5 else None

    metrics = {
        'epoch': epoch + 1,
        'test_loss': test_loss,
        'test_accuracy': accuracy,
        'test_accuracy_top5': top5_accuracy if calc_acc5 else None
    }
    metric_logger.log_metrics(metrics)

    step = epoch * len(train_dataloader.dataset)
    if writer is not None:
        writer.add_scalar('test loss', test_loss, step)
        writer.add_scalar('test accuracy', accuracy, step)
        if calc_acc5:
            writer.add_scalar('test accuracy5', top5_accuracy, step)

    logger.info(f"Test Results - Epoch {epoch+1}: Accuracy={accuracy:.2f}%, Avg loss={test_loss:.6f}")
    if calc_acc5:
        logger.info(f"Top-5 Accuracy={top5_accuracy:.2f}%")

if __name__ == "__main__":
    params = Params()
    
    # Create metric logger
    log_dir = os.path.join("logs", params.name)
    metric_logger = MetricLogger(log_dir)

    training_folder_name = '/mnt/imagenet/ILSVRC/Data/CLS-LOC/train'
    val_folder_name = '/mnt/imagenet/ILSVRC/Data/CLS-LOC/val'

    train_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        root=training_folder_name,
        transform=train_transformation
    )
    train_sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        sampler=train_sampler,
        num_workers=params.workers,
        pin_memory=True,
    )

    val_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=256, antialias=True),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = torchvision.datasets.ImageFolder(
        root=val_folder_name,
        transform=val_transformation
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32,
        num_workers=params.workers,
        shuffle=False,
        pin_memory=True
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using {device} device")

    resume_training = True

    num_classes = len(train_dataset.classes)
    model = ResNet50(num_classes=num_classes)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 
                               lr=params.max_lr/params.div_factor,
                               momentum=params.momentum,
                               weight_decay=params.weight_decay)

    # Initialize GradScaler for AMP
    scaler = GradScaler()

    steps_per_epoch = len(train_loader)
    total_steps = params.epochs * steps_per_epoch

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.175,
        total_steps=total_steps,
        pct_start=params.pct_start,
        div_factor=params.div_factor,
        final_div_factor=params.final_div_factor
    )

    start_epoch = 0
    checkpoint_path = os.path.join("checkpoints", params.name, f"checkpoint.pth")
    
    if resume_training and os.path.exists(checkpoint_path):
        print("Resuming training from checkpoint")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        assert params == checkpoint["params"]

    from torch.utils.tensorboard import SummaryWriter
    from pathlib import Path
    Path(os.path.join("checkpoints", params.name)).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter('runs/' + params.name)
    
    test(val_loader, model, loss_fn, epoch=0, writer=writer, train_dataloader=train_loader, 
         metric_logger=metric_logger, calc_acc5=True)
    
    print("Starting training")
    for epoch in range(start_epoch, params.epochs):
        print(f"Epoch {epoch}")
        train(train_loader, model, loss_fn, optimizer, epoch=epoch, writer=writer, 
              scaler=scaler, metric_logger=metric_logger)
        
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "params": params
        }
        torch.save(checkpoint, os.path.join("checkpoints", params.name, f"model_{epoch}.pth"))
        torch.save(checkpoint, os.path.join("checkpoints", params.name, f"checkpoint.pth"))
        
        test(val_loader, model, loss_fn, epoch + 1, writer, train_dataloader=train_loader,
             metric_logger=metric_logger, calc_acc5=True)
