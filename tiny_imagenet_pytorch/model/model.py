import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class TinyImageNetModel(nn.Module):

    def __init__(self):
        super(TinyImageNetModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(in_features=8 * 8 * 128, out_features=256)
        self.fcbn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(in_features=256, out_features=200)
        self.dropout_rate = 0.2

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(F.max_pool2d(x, 2))
        x = self.bn2(self.conv2(x))
        x = F.relu(F.max_pool2d(x, 2))
        x = self.bn3(self.conv3(x))
        x = F.relu(F.max_pool2d(x, 2))

        x = x.view(-1, 8 * 8 * 128)
        x = F.dropout(F.relu(self.fcbn1(self.fc1(x))), p=self.dropout_rate, training=self.training)
        x = self.fc2(x)
        y_pred = F.softmax(x, dim=-1)
        return y_pred

def accuracy(target, preds):
    preds_labels = np.argmax(preds, axis=1)
    return np.mean(target == preds_labels)

def fetch_metrics():
    metrics = {
        'accuracy': accuracy
    }
    return metrics
