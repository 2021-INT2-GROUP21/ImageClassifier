from torch import nn
import torch


class ModularClassifier(nn.Module):
    def __init__(self, small, medium, large):
        super().__init__()

        self.splits = small + medium + large
        self.small = small
        self.medium = medium
        self.large = large

        self.split_conv = nn.ModuleList()
        self.split_dense = nn.ModuleList()

        ####### SMALL ##################################################################################################

        for i in range(self.small):
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
                )
            )

        for i in range(self.small):
            self.split_dense.append(
                # Dense Layer
                nn.Sequential(
                    nn.Linear(32*16*16, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.05),
                    nn.Linear(512, 64),
                    nn.ReLU(inplace=True),
                ),
            )

        ####### LARGE ##################################################################################################

        for i in range(self.medium):
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
                )
            )

        for i in range(self.medium):
            self.split_dense.append(
                # Dense Layer
                nn.Sequential(
                    nn.Linear(64 * 8 * 8, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.05),
                    nn.Linear(512, 64),
                    nn.ReLU(inplace=True),
                ),
            )

        ####### LARGE ##################################################################################################

        for i in range(self.large):
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

        for i in range(self.large):
            self.split_dense.append(
                # Dense Layer
                nn.Sequential(
                    nn.Linear(128 * 4 * 4, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.05),
                    nn.Linear(512, 64),
                    nn.ReLU(inplace=True),
                ),
            )

        ################################################################################################################

        self.final_dense = nn.Sequential(
            nn.Linear(64 * self.splits, 10 * self.splits),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.08),
            nn.Linear(10 * self.splits, 10),
        )

    def forward(self, x):
        # Flatten images into vectors
        # conv layers
        xs = list()
        for i in range(self.splits):
            xs.append(self.split_conv[i](x))

        # flatten
        for i in range(self.splits):
            xs[i] = xs[i].view(xs[i].size(0), -1)

        # split dense
        for i in range(self.splits):
            xs[i] = self.split_dense[i](xs[i])

        # combine outputs from all splits and run in dense
        x = self.final_dense(torch.cat(tuple(xs), 1))

        return x