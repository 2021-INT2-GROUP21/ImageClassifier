import os

from torch import nn

from models.SplitClassifier import *
from models.TwoStep import *
from util import *


class Combined(nn.Module):
    def __init__(self):
        super().__init__()

        self.two_step = TwoStep()
        if os.path.isfile(get_save_path(self.two_step)):
            self.two_step.load_state_dict(torch.load(get_save_path(self.two_step)))

        self.split = SplitClassifier()
        if os.path.isfile(get_save_path(self.split)):
            self.split.load_state_dict(torch.load(get_save_path(self.split)))

        self.final_dense = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Flatten images into vectors
        # conv layers
        x = torch.cat((self.two_step(x), self.split(x)), 1)

        # flatten
        #x = x.view(x.size(0), -1)

        # fc layer
        x = self.final_dense(x)

        return x
