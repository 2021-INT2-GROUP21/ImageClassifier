from train import *
from test import *

from models.Combined import *


def main():
    train(SplitClassifier(5))

    #train(TwoStep())
    #test(TwoStep())

    #train(Combined())
    #test(Combined())


if __name__ == "__main__":
    main()
