import torch
import torch.optim as optim
import LoadData
from torch.optim.lr_scheduler import MultiStepLR
import wandb


def training(device, epochs, neurons, dropout, lr, l2, batch_size, milestones, name, model, train_path, train_txt_path, val_path, val_txt_path, exp_name, checkpoint_dir):

    net = model.Net(neurons, dropout)
    net.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()

    # set wandb options
    torch.manual_seed(23)
    wandb.init(project='micro_classification',
               config={"exp_name": exp_name, "learning_rate": lr,
                       "batch_size": batch_size,
                       "epochs": epochs, "milestones": milestones})
    wandb.run.name = exp_name + '_' + name

    # Vgg16
    if name == "Vgg16":
        optimizer = optim.Adam(net.net.classifier.parameters(), lr=lr, weight_decay=l2)
        scheduler = MultiStepLR(optimizer, milestones=[milestones], gamma=0.1)
    else:
        optimizer = optim.Adam(net.net.fc.parameters(), lr=lr, weight_decay=l2)
        scheduler = MultiStepLR(optimizer, milestones=[milestones], gamma=0.1)

    # Create dataset and data loader for training and validation
    traindata, valdata = LoadData.load_data(name, train_path, train_txt_path, val_path, val_txt_path, test_path=False,
                                            test_txt_path=False, extract=False)
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

        scheduler.step()

        # Validation loss
        val_loss_list = []
        total = 0
        correct = 0

        net.eval()  # set the network in evaluation mode

        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs.squeeze(0))

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
        print(f"train loss: ", sum(loss_list) / len(loss_list))
        print(f"train accuracy", correct_train / total_train)

        val_acc = correct / total
        if val_acc > best_val_acc:
            torch.save({
                'epoch': epoch + 1,  # because it starts from 0
                'model': net,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_list,
                'val_loss': val_loss_list,
            }, checkpoint_dir + exp_name + '_' + name + '.pt')
            print('finished epoch: saved model')
            best_val_acc = val_acc
            wandb.config.update({"best_val_acc": best_val_acc}, allow_val_change=True)
            wandb.config.update({"best_epoch": epoch + 1}, allow_val_change=True)

    print("Finished Training")
    print('Best_val_acc = ', best_val_acc)
