from CIFAR10_Model import CIFAR10Model
from util import *

from torch import optim
from torch import nn


def train():
    # create model
    device = get_device()
    model = CIFAR10Model().to(device)
    # load dataset
    train_loader, val_loader, classes = get_train_data()
    # define optimiser, Stochastic Gradient Descent
    params = model.parameters()
    optimiser = optim.SGD(params, lr=1e-2)
    # define loss TODO: I would prefer to use MSEloss instead of CEloss thus change the model to cater to a MSEloss
    #  training. Also if you want to use MSE loss, batch_size must equal model output size, 10. loss = nn.MSELoss()
    loss = nn.CrossEntropyLoss()

    # training and validation loop

    num_epochs = 16
    last_val_acc = 0

    for epoch in range(num_epochs):

        losses = list()
        accuracies = list()
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x)
            ce_loss = loss(out, y)
            model.zero_grad()
            ce_loss.backward()
            optimiser.step()

            losses.append(ce_loss.item())
            accuracies.append(y.eq(out.detach().argmax(dim=1)).float().mean())
        print(
            f'Epoch {epoch + 1},'
            f' train loss: {torch.tensor(losses).mean():.2f},'
            f' train acc: {torch.tensor(accuracies).mean():.2f}'
        )

        losses = list()
        accuracies = list()
        for batch in val_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            with torch.no_grad():
                out = model(x)
            ce_loss = loss(out, y)

            losses.append(ce_loss.item())
            accuracies.append(y.eq(out.detach().argmax(dim=1)).float().mean())
        current_val_acc = torch.tensor(accuracies).mean()
        print(
            f'Epoch {epoch + 1},'
            f' validation loss: {torch.tensor(losses).mean():.2f},'
            f' validation acc: {current_val_acc:.2f}'
        )

        if current_val_acc <= last_val_acc:
            break
        else:
            last_val_acc = current_val_acc
    print(f'Training complete in {epoch + 1} iteration with training accuracy of {100 * last_val_acc:.2f}%')

    torch.save(model.state_dict(), get_save_path(model))
    print("Trained model " + model.__class__.__name__ + " saved to " + get_save_path(model))


