import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

# # For Debug
import config

# class CNN(nn.Module):
#     def __init__(self, num_classes: int = 10, input_shape: tuple = (3, 224, 224)):
#         super().__init__()
#         self.feature_extractor = nn.Sequential(
#             nn.Conv2d(3, 6, kernel_size=7, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(6),
#             nn.MaxPool2d(kernel_size=3, stride=1),
#             nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(6),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.AdaptiveAvgPool2d((5, 5))  # Ensure (16, 5, 5) output
#         )

#         # Compute number of features dynamically
#         with torch.no_grad():
#             dummy_input = torch.zeros(1, *input_shape)
#             output = self.feature_extractor(dummy_input)
#             num_features = output.view(1, -1).size(1)  # E.g., 16 * 5 * 5 = 400

#         self.classifier = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.Linear(num_features, 120),
#             nn.ReLU(),
#             nn.Linear(120, 84),
#             nn.ReLU(),
#             nn.Linear(84, num_classes),
#         )

#     def forward(self, input: Tensor) -> Tensor:
#         x = self.feature_extractor(input)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 1)
        self.pool = nn.MaxPool2d((2, 2))
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(32 * 28 * 28, 120) # input_channels, output_channels
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4) # 4 cat classes
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # x = self.pool(F.relu(self.bn4(self.conv4(x))))
        # x = self.pool(F.relu(self.bn5(self.conv5(x))))
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        # x = self.pool(F.relu(self.conv4(x)))
        # x = self.pool(F.relu(self.conv5(x)))
        # x = self.pool(F.relu(self.conv6(x)))
        x = torch.flatten(x, 1)
        # x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output

class AlexNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.feature_extractor = nn.Sequential(  
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(96, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=(256 * 2 * 2), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# # For Debug
# model = CNN().to(config.DEVICE)
# print(model)
# model = AlexNet(config.NUM_CLASSES).to(config.DEVICE)
# print(model)
