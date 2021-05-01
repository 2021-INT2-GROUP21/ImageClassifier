from train import *
from test import *

from models.Combined import *


def main():
    while True:
        for i in range(1, 10, 1):
            train(SplitClassifier(i))

    #train(TwoStep())
    #test(TwoStep())

    #train(Combined())
    #test(Combined())


if __name__ == "__main__":
    main()
