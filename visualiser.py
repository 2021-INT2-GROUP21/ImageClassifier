from models.SplitClassifier import SplitClassifier
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

    model = SplitClassifier()

    writer.add_graph(model, images)
    writer.close()

    batch_size=32
    torchinfo.summary(model, input_size=(batch_size, 3, 32, 32))


if __name__ == '__main__':
    main(SplitClassifier())
