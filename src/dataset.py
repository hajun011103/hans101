import os
import natsort
from PIL import Image
import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset
# For Debug
import numpy as np
import pandas as pd
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import config

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, class_list, transform=None):
        self.root_dir = root_dir
        self.class_list = class_list
        self.transform = transform
        self.samples = []
        self.label = {}

        for class_count, class_name in enumerate(class_list):
            if os.path.exists(os.path.join(root_dir, class_name)):
                class_dir = os.path.join(root_dir, class_name)
                for img_idx in natsort.natsorted(os.listdir(class_dir)):
                    img_path = os.path.join(class_dir, img_idx)
                    self.samples.append((img_path, class_count))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def GetTransforms(train:True):
    if train: # Train
        transform = v2.Compose([
            v2.PILToTensor(),
            v2.Resize((224, 224)),
            # v2.RandomApply([ # Randomly apply below transformation
            # v2.RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(degrees=10),
            # v2.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
            # v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else: # Validation & Test
        transform = v2.Compose([
            v2.PILToTensor(),
            v2.Resize((224, 224)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    return transform

def LoadData():
    train_dataset = CustomImageDataset(config.TRAIN_DIR, config.CLASS_LIST, transform=GetTransforms(True))
    val_dataset = CustomImageDataset(config.TEST_DIR, config.CLASS_LIST, transform=GetTransforms(False))

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=8)

    # CutMix & MixUp
    cutmix = v2.CutMix(num_classes=config.NUM_CLASSES)
    mixup = v2.MixUp(num_classes=config.NUM_CLASSES)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    return train_loader, val_loader, cutmix_or_mixup

# # For Debug
# if __name__ == "__main__":
#     train_loader, val_loader, cutmix_or_mixup = LoadData()



#     # for images, labels in train_loader:
#     #     print(f"{images.shape = }, {labels.shape = }")
#     #     print(labels.dtype)
#     #     break

#     def imshow(img):
#         img = img / 2 + 0.5
#         npimg = img.numpy()
#         plt.imshow(np.transpose(npimg, (1, 2, 0)))
#         plt.show()

#     dataiter = iter(train_loader)
#     # image, label = cutmix_or_mixup(next(dataiter))
#     image, label = next(dataiter)
#     imshow(torchvision.utils.make_grid(image))
#     print(' '.join(f'{config.CLASS_LIST[label[j]]:5s}' for j in range(config.BATCH_SIZE)))

def ArrangeDataset(): # For re-indexing the train images
    # Train dataset    
    # for cat in config.CLASS_LIST:
    #     directory = os.path.join(config.TRAIN_DIR + cat + "/")
    #     print(directory)
    #     files = natsort.natsorted(os.listdir(directory))
    #     print(files)
    #     print(len(files))
    #     for idx in range(0, len(files)):
    #         os.rename(directory + files[idx], directory + f"{cat}{idx}.jpg")

    # Label Test dataset
    # label = "ssttrttnrrrrsntsntrnsn"
    # label = list(label)
    # print(label)
    # print(type(label))
    # directory = os.path.join(config.TEST_DIR)
    # print(directory)
    # files = natsort.natsorted(os.listdir(directory))
    # print(files)
    # for idx, cat in enumerate(label):
    #     print(cat, idx)
    #     if cat == "s":
    #        cat = "SiameseCat"
    #     if cat== "t":
    #        cat = "TuxedoCat"
    #     if cat== "n":
    #        cat = "NorwegianForestCat"
    #     if cat== "r":
    #         cat = "RussianBlueCat"
    #     print(cat)
    #     os.rename(directory + files[idx], directory + cat + f"{idx}.jpg")
    for cat in config.CLASS_LIST:
        directory = os.path.join(config.TEST_DIR + cat + "/")
        print(directory)
        files = natsort.natsorted(os.listdir(directory))
        print(files)
        print(len(files))
        for idx in range(0, len(files)):
            os.rename(directory + files[idx], directory + f"{cat}{idx}.jpg")


# ArrangeDataset()

        