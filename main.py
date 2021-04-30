from train import *
from test import *

from models.TripleConvBlocks import *
from models.SingleConvLayer import *


def main():
    train(CIFAR10Model())
    test(CIFAR10Model())

    #train(SingleConvLayer())
    #test(SingleConvLayer())


if __name__ == "__main__":
    main()
