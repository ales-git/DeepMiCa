import os.path
import torch
import train as t
import random
import wandb

def main(train_path, train_txt_path, val_path, val_txt_path, checkpoint_dir):
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # Choose the pretrained model to use
    model_name=0
    while model_name not in [1,2]:

        model_name = int(input("Choose the pretrained network to use (1 = Vgg16, 2 = Resnet18):\n"))

        if model_name == 1:
            import Vgg16 as model
            pretrained_model = "Vgg16"

        elif model_name == 2:
            import Resnet18 as model
            pretrained_model = "Resnet18"

        else:
            print("Invalid enter.")

    ######################### TRAINING #########################

    epochs = 300
    batch_size = 64
    lr = 1e-4
    neurons = 128
    dropout = 0.3
    l2 = 1e-2
    milestones = 60
    exp_name = "classification_experiment"

    t.training(device, epochs, neurons, dropout, lr, l2, batch_size, milestones, pretrained_model, model,
               train_path, train_txt_path, val_path, val_txt_path, exp_name, checkpoint_dir)

    wandb.finish()
    ############################################################


if __name__=="__main__":
    random.seed(23)

    # Define paths
    main_path = '../_02_SplitData/CBIS_DDSM/'

    train_path = main_path + 'train_img/'
    train_txt_path = main_path + 'txt/train.txt'
    val_path = main_path + 'val_img/'
    val_txt_path = main_path + 'txt/validation.txt'

    checkpoint_dir = 'Checkpoints/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    main(train_path, train_txt_path, val_path, val_txt_path, checkpoint_dir)