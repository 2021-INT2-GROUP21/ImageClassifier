from models.TripleConvBlocks import CIFAR10Model
from util import *
from torch.utils.tensorboard import SummaryWriter
import torchinfo

'''
In command line run tensorboard --logdir=<FilePath to ./ImageClassifier>
run this file and go to the address given.
'''


def main(in_model):
    model = in_model
    writer = SummaryWriter('runs/' + model.__class__.__name__)

    train_loader, val_loader, classes = get_train_data()

    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    model = in_model

    writer.add_graph(model, images)
    writer.close()

    batch_size=32
    torchinfo.summary(model, input_size=(batch_size, 3, 32, 32))
    plt_graph()


def plt_graph():
    import matplotlib.pyplot as plt
    import csv
    import numpy as np

    with open('datapoints.csv', newline="") as f:
        reader = csv.reader(f)
        data = list(reader)

    taccuracy = list(map(float, data[0][1:]))
    tloss = list(map(float, data[1][1:]))
    vaccuracy = list(map(float, data[2][1:]))
    vloss = list(map(float, data[3][1:]))
    x = list(range(1, len(taccuracy) + 1))

    lossplot = plt.figure(1, figsize=(8, 8))
    plt.title("Loss Over Epoch")
    plt.plot(x, tloss, label="training loss")
    plt.plot(x, vloss, label="validation loss")
    plt.yticks(np.arange(0, 1.2, step=0.1))
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    accloss = plt.figure(2, figsize=(8, 8))
    plt.title("Accuracy Over Epoch")
    plt.plot(x, taccuracy, label="training accuracy")
    plt.plot(x, vaccuracy, label="validation accuracy")
    plt.yticks(np.arange(0.55, 1, step=0.05))
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()

    plt.show()


if __name__ == '__main__':
    #plt_graph()
    main(CIFAR10Model())
