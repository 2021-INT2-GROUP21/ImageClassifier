import os

from util import *


def test(input_model):
    test_loader, classes = get_test_data()

    # create model
    device = get_device()
    trained_model = input_model.to(device)
    trained_model.load_state_dict(torch.load(get_save_path(trained_model)))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = trained_model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    with open(get_save_path(trained_model) + ".txt", "w") as f:
        for i in range(10):
            # print('Accuracy of %10s : %2d %%' % (
            #    classes[i], 100 * class_correct[i] / class_total[i]))
            f.write('Accuracy of %10s : %2d %%\n' % (
                classes[i], 100 * class_correct[i] / class_total[i]))

        print('Accuracy : %2d %%' % (
                100 * sum(class_correct) / sum(class_total)))
        f.write('\nAccuracy of %10s : %2d %%' % (
            "total", 100 * sum(class_correct) / sum(class_total)))

    if not os.path.isfile(get_save_path(trained_model) + ".csv"):
        with open(get_save_path(trained_model) + ".csv", "a") as f:
            f.write("plane,car,bird,cat,deer,dog,frog,horse,ship,truck,total")

    with open(get_save_path(trained_model) + ".csv", "a") as f:
        for i in range(10):
            f.write(str(100 * class_correct[i] / class_total[i]) + ",")

        f.write(str(100 * sum(class_correct) / sum(class_total)) + ",\n")
