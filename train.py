import os

from util import *
from test import *

from torch import optim
from torch import nn


def train(input_model):
    # create model
    device = get_device()
    model = input_model.to(device)
    #if os.path.isfile(get_save_path(model)):
    #    model.load_state_dict(torch.load(get_save_path(model)))
    # load dataset
    train_loader, val_loader, classes = get_train_data()
    # define optimiser, Stochastic Gradient Descent
    params = model.parameters()
    lr = 1e-2
    optimiser = optim.SGD(params, lr=lr)

    loss = nn.CrossEntropyLoss()

    # training and validation loop

    num_epochs = 32
    last_val_acc = 0

    plt_epoch_tloss = ['tloss']
    plt_epoch_taccuracy = ['taccuracy']
    plt_epoch_vloss = ['vloss']
    plt_epoch_vaccuracy = ['vaccuracy']

    print(f'Starting training with parameters: lr={lr}, num epochs={num_epochs}')

    for epoch in range(num_epochs):

        losses = list()
        accuracies = list()
        for i, batch in enumerate(train_loader, 0):
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x)
            ce_loss = loss(out, y)
            model.zero_grad()
            ce_loss.backward()
            optimiser.step()

            losses.append(ce_loss.item())
            accuracies.append(y.eq(out.detach().argmax(dim=1)).float().mean())
            # for some reason pycharm doesn't support \r so this only shows in a terminal
            # TODO: Make this a cleaner solution to seeing progress
            print("progress: " + str(int((i / len(train_loader)) * 100)) + "%", end='\r')
        print(
            f'Epoch {epoch + 1},'
            f' train loss: {torch.tensor(losses).mean():.2f},'
            f' train acc: {torch.tensor(accuracies).mean():.2f}'
        )

        plt_epoch_tloss.append(float(torch.tensor(losses).mean()))
        plt_epoch_taccuracy.append(float(torch.tensor(accuracies).mean()))

        losses = []
        accuracies = []

        for batch in val_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            with torch.no_grad():
                out = model(x)
            ce_loss = loss(out, y)

            losses.append(ce_loss.item())
            accuracies.append(y.eq(out.detach().argmax(dim=1)).float().mean())

        current_val_acc = round(float(torch.tensor(accuracies).mean()), 2)
        print(
            f'Epoch {epoch + 1},'
            f' validation loss: {torch.tensor(losses).mean():.2f},'
            f' validation acc: {current_val_acc:.2f}'
        )

        plt_epoch_vloss.append(float(torch.tensor(losses).mean()))
        plt_epoch_vaccuracy.append(float(torch.tensor(accuracies).mean()))

        torch.save(model.state_dict(), get_save_path(model))
        test(model)

        if current_val_acc <= last_val_acc or lr <= 1e-4:
            break
        else:
            if ((current_val_acc - last_val_acc)/current_val_acc) <= lr*10:
                lr = lr * 1e-1
                optimiser = optim.SGD(model.parameters(), lr=lr)
                print(f'optimiser lr={lr}')
            last_val_acc = current_val_acc
    print(
        f'Training complete in {epoch + 1}'
        f' iteration with training accuracy of {100 * last_val_acc:.2f}%'
        )

    #redundant saving if it is being saved to call the test
    #torch.save(model.state_dict(), get_save_path(model))
    print("Trained model " + model.__class__.__name__ + " saved to " + get_save_path(model))

    import csv
    with open("datapoints.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([plt_epoch_taccuracy,plt_epoch_tloss,plt_epoch_vaccuracy,plt_epoch_vloss])
    from visualiser import plt_graph
    plt_graph()
