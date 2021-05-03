import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


class TwoStep(nn.Module):
    def __init__(self):
        super().__init__()
        m = 32

        #######################################################################################

        self.conv_layer0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=m, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(m),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=m, out_channels=2 * m, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(2 * m),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(in_channels=2 * m, out_channels=3 * m, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(3 * m),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=3 * m, out_channels=4 * m, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(4 * m),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.5),
        )

        #######################################################################################

        #self.conv_layer1 = nn.Sequential(
        #    nn.Conv2d(in_channels=4*m, out_channels=6*m, kernel_size=(3, 3), padding=(1, 1)),
        #    nn.BatchNorm2d(6*m),
        #    nn.ReLU(inplace=True),
        #    nn.Conv2d(in_channels=6 * m, out_channels=8 * m, kernel_size=(5, 5), padding=(2, 2)),
        #    nn.BatchNorm2d(8 * m),
        #    nn.ReLU(inplace=True),
        #    nn.MaxPool2d(kernel_size=2, stride=2),
        #    nn.Dropout2d(p=0.5),
        #)

        #######################################################################################

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.5),
            # 2 8x8 images with 12*m channels
            nn.Linear(8 * 8 * 4*m, 2048),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=0.1),
            #nn.Linear(4096, 1024),
            #nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Flatten images into vectors
        # conv layers
        x = self.conv_layer0(x)
        #x = self.conv_layer1(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)

        return x
