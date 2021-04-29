from torch import nn
import torch


class SplitClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.split_conv = nn.ModuleList()
        self.split_dense = nn.ModuleList()

        for i in range(10):
            self.split_conv.append(
                nn.Sequential(
                    # Conv Layer block 1
                    nn.Sequential(
                        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=(1, 1)),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Dropout2d(p=0.01),
                    ),
                    # Conv Layer block 2
                    nn.Sequential(
                        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Dropout2d(p=0.05),
                    ),
                    # Conv Layer block 3
                    nn.Sequential(
                        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Dropout2d(p=0.03),
                    ),
                )
            )

        for i in range(10):
            self.split_dense.append(
                # Dense Layer
                nn.Sequential(
                    nn.Linear(128*4*4, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.05),
                    nn.Linear(512, 64),
                    nn.ReLU(inplace=True),
                ),
            )

        self.final_dense = nn.Sequential(
            nn.Linear(640, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.08),
            nn.Linear(100, 10),
        )

    def forward(self, x):
        # Flatten images into vectors
        # conv layers
        xs = list()
        for i in range(10):
            xs.append(self.split_conv[i](x))

        # flatten
        for i in range(10):
            xs[i] = xs[i].view(xs[i].size(0), -1)

        # split dense
        for i in range(10):
            xs[i] = self.split_dense[i](xs[i])

        # combine outputs from all splits and run in dense
        x = self.final_dense(torch.cat(tuple(xs), 1))

        return x
