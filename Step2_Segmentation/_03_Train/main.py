import torch
import train as t
import UNet as model
import random
import wandb
import os

if __name__ == '__main__':
    # check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    random.seed(0)

    # Choose the dataset to use
    dataset = 0
    while dataset not in [1, 2]:

        dataset = int(
            input(
                "Choose if you want to train the model on train/val/test or the entire dataset (1 = train/val/test, 2 = entire dataset):\n"))

        if dataset == 1:
            flag_trainAll = False
            # train and val patches are in the same folder train_val
            main_path = "../_02_Patches/INbreast/Patches_dataset"
            path_train_img = os.path.join(main_path, "train_val_img")
            path_train_mask = os.path.join(main_path, "train_val_mask")

            epochs = 300

        elif dataset == 2:
            flag_trainAll = True
            ''' You should adjust the paths and code accordingly '''
            print('Please adjust the paths and code accordingly.')
        else:
            print("Invalid enter.")

    if dataset == 1:
        ######################### TRAINING #########################
        # params
        batch_size = 8
        lr = 0.001
        milestones = 100

        for i in range(1, 11):
            exp_name = 'FOLD_' + str(i)
            file_train_k = '../_01_SplitData/INbreast/txt_cv/train' + str(i) + '.txt'
            file_val_k = '../_01_SplitData/INbreast/txt_cv/validation' + str(i) + '.txt'

            t.training(device, file_train_k, file_val_k,
                       epochs, lr, milestones,
                       batch_size, model,
                       path_train_img, path_train_mask,
                       exp_name, flag_trainAll)

            wandb.finish()
