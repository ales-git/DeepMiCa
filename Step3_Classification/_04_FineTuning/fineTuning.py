import os.path
import sys
import torch
import torch.optim as optim
# import Step3_Classification._03_FeatureExtraction.LoadData as LoadData
import wandb

sys.path.append('../_03_FeatureExtraction')
import LoadData as LoadData
import Resnet18
import Vgg16


def finetuning(train_path, train_txt_path, val_path, val_txt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # Choose the pretrained model to use
    model_name = 0
    while model_name not in [1, 2]:

        model_name = int(
            input("Choose the pretrained network to use (1 = Vgg16, 2 = Resnet18):\n"))

        if model_name == 1:
            net_name = "Vgg16"

        elif model_name == 2:
            net_name = "Resnet18"

        else:
            print("Invalid enter.")

    exp_name = 'classification_experiment_FT'
    checkpoint_dir = 'Checkpoints/'
    if os.path.exists(checkpoint_dir) == False:
        os.makedirs(checkpoint_dir)
    path_checkp_FE = '../_03_FeatureExtraction/Checkpoints/classification_experiment_' + net_name + '.pt'
    checkpoint = torch.load(path_checkp_FE, map_location=torch.device(device))
    net = checkpoint['model']
    print(net)
    net.load_state_dict(checkpoint['model_state_dict'])

    # Parameters
    epochs = 150
    batch_size = 64
    lr = 1e-05
    l2 = checkpoint['optimizer_state_dict']['param_groups'][0]['weight_decay']

    # set wandb options
    torch.manual_seed(23)
    wandb.init(project='micro_classification',
               config={"exp_name": exp_name, "learning_rate": lr,
                       "batch_size": batch_size,
                       "epochs": epochs, "lr": lr, "l2": l2})
    wandb.run.name = exp_name + '_' + net_name

    def unfrozen_layers(name, child, list_unfrozen):

        if name in list_unfrozen:
            print(name + ' is unfrozen')
            for param in child.parameters():
                param.requires_grad = True
        else:
            print(name + ' is frozen')
            for param in child.parameters():
                param.requires_grad = False

    # VGG16
    if net.net.__class__.__name__ == 'VGG':
        layer4 = ['24', '25', '26', '27', '28', '29', '30', '31', '32', '33']
        layer5 = ['34', '35', '36', '37', '38', '39', '40', '41', '42', '43']
        classifier = ['0', '1', '2', '3', '4']

        for name, child in net.net.features.named_children():
            unfrozen_layers(name, child, layer5)  # layer5+layer4

        for name, child in net.net.classifier.named_children():
            unfrozen_layers(name, child, classifier)

    # ResNet18
    elif net.net.__class__.__name__ == 'ResNet':
        for name, child in net.net.named_children():
            unfrozen_layers(name, child, ['layer4', 'fc'])

    net.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.net.parameters()), lr=lr, weight_decay=l2)

    traindata, valdata = LoadData.load_data(net_name, train_path, train_txt_path, val_path, val_txt_path)
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=batch_size, shuffle=False)

    ### Train Loop
    best_val_acc = 0.0

    for epoch in range(epochs):  # loop over the dataset multiple times
        print(f"Epoch {epoch + 1}\n-------------------------------")
        loss_list = []
        total_train = 0
        correct_train = 0

        net.train()  # set the network in training mode

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.squeeze(0))
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

            # print statistics
            loss_list.append(loss.item())

            # output predictions on training set
            predicted = torch.round(torch.nn.Sigmoid()(outputs.data))
            total_train += labels.size(0)
            correct_train += (predicted.squeeze(1) == labels).sum().item()

        # Validation loss
        val_loss_list = []
        total = 0
        correct = 0

        net.eval()  # set the network in evaluation mode

        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs.squeeze())

                predicted = torch.round(torch.nn.Sigmoid()(outputs.data))
                total += labels.size(0)
                correct += (predicted.squeeze(1) == labels).sum().item()

                val_loss = criterion(outputs, labels.unsqueeze(1).float())
                val_loss_list.append(val_loss.item())

        wandb.log({"train loss": sum(loss_list) / len(loss_list),
                   "val loss": sum(val_loss_list) / len(val_loss_list),
                   "train accuracy": correct_train / total_train,
                   "val accuracy": correct / total})

        print(f"val loss: ", sum(val_loss_list) / len(val_loss_list))
        print(f"val accuracy: ", correct / total)

        # Save model
        val_acc = correct / total
        if val_acc > best_val_acc:
            torch.save({
                'epoch': epoch + 1,  # because it starts from 0
                'model': net,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_list,
                'val_loss': val_loss_list,
            }, checkpoint_dir + '/' + exp_name + '_' + net_name + '.pt')
            print('finished epoch: saved model')
            best_val_acc = val_acc

    print("Finished Training")
    print('Best_val_acc = ', best_val_acc)


if __name__ == '__main__':
    main_path = '../_02_SplitData/CBIS_DDSM/'
    train_path = main_path + 'train_img/'
    train_txt_path = main_path + 'txt/train.txt'
    val_path = main_path + 'val_img/'
    val_txt_path = main_path + 'txt/validation.txt'

    finetuning(train_path, train_txt_path, val_path, val_txt_path)
