import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import LoadData
import os
from torchmetrics.classification import MulticlassJaccardIndex as IoU
from Losses import topk_CE
from datetime import datetime
import wandb


def training(device, file_train_k, file_val_k, epochs, lr, milestones, batch_size, model, path_train_img,
             path_train_mask, exp_name, flag_trainAll):

    # set wandb options
    torch.manual_seed(23)
    wandb.init(project='micro_segmentation',
               config={"exp_name": exp_name, "learning_rate": lr,
                       "batch_size": batch_size,
                       "epochs": epochs, "milestones": milestones})
    wandb.run.name = exp_name

    # Define net
    net = model.build_unet()
    net.to(device)

    ###############################

    # Define criterion and optimizer
    criterion = topk_CE()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.99)
    scheduler = MultiStepLR(optimizer, milestones=[milestones], gamma=0.1)

    ###############################

    # Define dataset and data loader
    def read_files_txt(file, path_set_img, path_set_mask):
        f = open(file, "r")
        patches_names, imgs, masks = [], [], []
        for line in f:
            line = line.strip()
            path_file, path_mask = line.split(" ")[0], line.split(" ")[1]
            complete_img_name = path_file.split('/')[-1].split('.')[0]
            # file names of all the patches from the selected image:
            list_all_train_val_patches = os.listdir(path_set_img)
            for patch_file in list_all_train_val_patches:
                if patch_file.startswith(complete_img_name):
                    imgs.append(path_set_img + '/' + patch_file)
                    masks.append(path_set_mask + '/' + patch_file)
        f.close()

        return imgs, masks

    inputs_train, targets_train = read_files_txt(file_train_k, path_train_img, path_train_mask)
    inputs_val, targets_val = read_files_txt(file_val_k, path_train_img, path_train_mask)

    if flag_trainAll == True:
        ''' You should adjust the paths and code accordingly '''
        print('You should adjust the paths to train on the whole dataset')
    else:
        traindata, valdata = LoadData.load_data(inputs_train, targets_train, inputs_val, targets_val)
        trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True)
        valloader = torch.utils.data.DataLoader(valdata, batch_size=batch_size, shuffle=False)

    ###############################

    # wandb.watch(net, log_freq=100)

    # Train Loop

    best_IoU = 0.0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loss_list = []
        print('Training...')

        net.train()  # set the network in training mode

        for i, (x, y) in enumerate(trainloader, 0):
            # get input image and mask
            image, target = x.squeeze(1).to(device), y.squeeze(1).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = net(image)  # forward pass
            loss = criterion(output, target).to(device)
            train_loss_list.append(loss.item())
            loss.backward()  # backward pass
            optimizer.step()  # update the parameters

            output = torch.round(torch.nn.Sigmoid()(output))

        scheduler.step()

        if flag_trainAll == False:
            print('Validation...')
            # evaluation
            net.eval()
            val_loss_list = []
            IoU_mean = []

            for i, (x, y) in enumerate(valloader, 0):
                with torch.no_grad():
                    image, target = x.squeeze(1).to(device), y.squeeze(1).to(device)
                    output = net(image)
                    loss = criterion(output, target).to(device)
                    val_loss_list.append(loss.item())
                    # binarize output mask
                    output = torch.round(torch.nn.Sigmoid()(output))  # output values from 0 to 1
                    # Compute IOU
                    output, target = output.cpu().detach(), target.cpu().detach()
                    for index in range(len(output)):  # loop on single batch images
                        compute_IoU = IoU(num_classes=2)
                        IoU_mean.append(compute_IoU(output[index].int(), target[index].int()))

            wandb.log({'epoch': epoch + 1,
                       'IoU': sum(IoU_mean) / len(IoU_mean),
                       'loss_val': sum(val_loss_list) / len(val_loss_list)})

            print(f"val loss: ", sum(val_loss_list) / len(val_loss_list))
            print(f"IoU val: ", sum(IoU_mean) / len(IoU_mean))

            IoU_val = sum(IoU_mean) / len(IoU_mean)

            if IoU_val > best_IoU:
                torch.save({
                    'epoch': epoch + 1,
                    'model': net,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss_list,
                    'val_loss': val_loss_list,
                }, '../Checkpoints/' + exp_name + '.pt')
                print('finished epoch: saved best model')
                best_IoU = IoU_val
                wandb.config.update({"best_epoch": epoch + 1}, allow_val_change=True)
            else:
                print('finished epoch')

        else:
            torch.save({
                'epoch': epoch + 1,
                'model': net,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss_list
            }, '../Checkpoints/' + exp_name + '.pt')
            print('finished epoch: saved best model')

        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print("date and time = ", dt_string)

    print("Finished Training")
    print('Best_IoU = ', best_IoU)
    wandb.config.update({"best_val_acc": best_IoU}, allow_val_change=True)
