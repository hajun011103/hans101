import os
import torch

TRAIN_DIR = "/mnt/new/HANS101_2025/Cat_Train_Dataset/"
TEST_DIR = "/mnt/new/HANS101_2025/Cat_Test_Dataset/"
SAVE_DIR = "/mnt/new/HANS101_2025/Models/"
SEED = 42
DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

CLASS_LIST = ("SiameseCat", "TuxedoCat", "NorwegianForestCat", "RussianBlueCat")
NUM_CLASSES = len(CLASS_LIST)

BATCH_SIZE = 64
NUM_EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
MIN_LR = 1e-15
PATIENCE = 1
