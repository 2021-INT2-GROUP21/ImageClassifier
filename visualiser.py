from models.CIFAR10_Model import CIFAR10Model
from util import *
from torch.utils.tensorboard import SummaryWriter

'''
In command line run tensorboard --logdir=<FilePath to ./ImageClassifier>
run this file and go to the address given.
'''


def getWriter():
    return SummaryWriter('runs/CIFAR10_Model')


def main():
    writer = getWriter()

    train_loader, val_loader, classes = get_train_data()

    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    model = CIFAR10Model()

    writer.add_graph(model, images)
    writer.close()


if __name__ == '__main__':
    main()
