import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import config
from dataset import LoadData
from model import CNN, AlexNet

def train(train_loader, val_loader):
    # For tensorboard Usage: tensorboard --logdir src/runs/
    writer = SummaryWriter()

    model = CNN().to(config.DEVICE)
    # model = AlexNet(config.NUM_CLASSES).to(config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr = config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=config.PATIENCE, 
                                  threshold=0.01, threshold_mode='rel', min_lr=config.MIN_LR)

    for epoch in range(1, config.NUM_EPOCHS+1):
        # Train
        model.train()
        train_loss = 0.0

        for image, label in tqdm(train_loader, desc=f"Train Epoch{epoch}"):
            # image, label = cutmix_or_mixup(image, label)
            image, label = image.to(config.DEVICE), label.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(image)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for image, label in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
                image, label = image.to(config.DEVICE), label.to(config.DEVICE)
                outputs = model(image)
                loss = criterion(outputs, label)
                val_loss += loss.item()
                # Accuracy
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == label).sum().item()
                val_total += label.size(0)

            val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\n[Epoch{epoch}/{config.NUM_EPOCHS}] ",
              f"Train Loss: {train_loss:.4f} | ",
              f"Val Acc: {val_acc:.3f}%, Loss: {val_loss:.4f} | ",
              f"LR: {current_lr:.8f}")
        
        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/Accuracy", val_acc, epoch)
        writer.add_scalar("Learning Rate", current_lr, epoch)
        scheduler.step(val_loss)

    torch.save(model.state_dict(), os.path.join(config.SAVE_DIR, "model.pth"))
    writer.close()


if __name__ == "__main__":
    train_loader, val_loader, cutmix_or_mixup = LoadData()
    train(train_loader, val_loader)

