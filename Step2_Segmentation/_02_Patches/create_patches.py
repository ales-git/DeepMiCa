from skimage.io import imread
import cv2
import os
from torchvision import transforms
from torchvision.io import read_image, write_png
import shutil
import pandas as pd
import sys
from patchify import patchify, unpatchify
import numpy as np


# The required condition to successfully recover the image
# using unpatchify is to have:
# (width - patch_width) mod step_size = 0 when calling patchify.

def prepare_folder(dataset_name, path_out):
    if dataset_name == 1:
        if not os.path.exists(path_out + 'train_val_img'):
            os.makedirs(path_out + 'train_val_img')

        if not os.path.exists(path_out + 'train_val_mask'):
            os.makedirs(path_out + 'train_val_mask')

        if not os.path.exists(path_out + 'test_img'):
            os.makedirs(path_out + 'test_img')

        if not os.path.exists(path_out + 'test_mask'):
            os.makedirs(path_out + 'test_mask')
    else:
        if not os.path.exists(path_out + 'AllPng'):
            os.makedirs(path_out + 'AllPng')

        if not os.path.exists(path_out + 'Masks'):
            os.mkdir(path_out + 'Masks')


def padding(list_img, list_mask, patch_size):
    for path_i, path_m in zip(list_img, list_mask):
        # image
        im = read_image(path_i)
        x = 0
        y = 0
        while (im.shape[1] + y) % patch_size != 0:
            y += 1
        while (im.shape[2] + x) % patch_size != 0:
            x += 1

        padding = transforms.Pad(padding=[0, 0, x, y])
        im = padding(im)

        mask = read_image(path_m)
        x = 0
        y = 0
        while (mask.shape[1] + y) % patch_size != 0:
            y += 1
        while (mask.shape[2] + x) % patch_size != 0:
            x += 1

        padding = transforms.Pad(padding=[0, 0, x, y])
        mask = padding(mask)

        write_png(im, path_i)
        write_png(mask, path_m)


def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)


def save_patches(list_img, list_mask, main_path_out, pw, dataset_name):
    prepare_folder(dataset_name, main_path_out)

    for img, mask in zip(list_img, list_mask):
        if dataset_name == 1:

            patlist_test = []
            with open('../_01_SplitData/INbreast/txt_cv/test.txt', 'r') as f:
                # read one row at a time
                for line in f:
                    pat = line.split('/')[4].split('.')[0]
                    patlist_test.append(str(pat))

            if img.split('/')[-1].split('.')[0] in patlist_test:
                path_out_img = main_path_out + 'test_img/'
                path_out_mask = main_path_out + 'test_mask/'
            else:
                path_out_img = main_path_out + 'train_val_img/'
                path_out_mask = main_path_out + 'train_val_mask/'

        else:
            path_out_img = main_path_out + 'AllPng/'
            path_out_mask = main_path_out + 'Masks/'

        large_image = imread(img, cv2.COLOR_BGR2GRAY)
        large_mask = imread(mask, cv2.COLOR_BGR2GRAY)  # mask has the same name

        patch_width = pw

        diff_w = large_image.shape[1] - patch_width
        diff_h = large_image.shape[0] - patch_width

        step_size = patch_width

        while (diff_w % step_size != 0 or diff_h % step_size != 0):
            step_size = step_size - 1

        print('step size: ' + str(step_size) + ' img: ' + img)

        patches_img = patchify(large_image, (patch_width, patch_width), step=step_size)
        patches_mask = patchify(large_mask, (patch_width, patch_width), step=step_size)

        if patches_img.max() <= 1.0:
            patches_img *= 255
            patches_mask *= 255

        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img = patches_img[i, j, :, :]
                single_patch_mask = patches_mask[i, j, :, :]
                cv2.imwrite(
                    path_out_img + (img.split('/')[-1]).split('.')[0] + '_' + str(i) + '_' + str(j) + '.png',
                    single_patch_img)
                if dataset_name == 1:
                    cv2.imwrite(
                        path_out_mask + (img.split('/')[-1]).split('.')[0] + '_' + str(i) + '_' + str(j) + '.png',
                        single_patch_mask)
                else:
                    cv2.imwrite(
                        path_out_mask + (mask.split('/')[-1]).split('.')[0] + '_' + str(i) + '_' + str(j) + '.png',
                        single_patch_mask)


if __name__ == '__main__':
    # Choose the dataset to use
    dataset_name = 0
    while dataset_name not in [1, 2]:

        dataset_name = int(
            input("Choose the dataset to use (1 = INbreast, 2 = CBIS-DDSM):\n"))

        if dataset_name == 1:

            main_path = '../../Step1_Preprocessing/INbreast/'

            # output
            main_path_out = 'INbreast/Patches_dataset/'

            # input
            path_img = main_path + 'AllPng/'
            path_mask = main_path + 'Masks/'

            list_img = [path_img + f for f in os.listdir(path_img) if not f.startswith('.')]
            list_mask = [path_mask + f for f in os.listdir(path_mask) if not f.startswith('.')]



        elif dataset_name == 2:

            main_path = '../../Step1_Preprocessing/CBIS_DDSM/'
            # input
            path_img = main_path + 'AllPng/'
            path_mask = main_path + 'Masks/'
            # output
            main_path_out = 'CBIS_DDSM/Patches_dataset/'

            path_csv = '../../Datasets/CBIS_DDSM/csv/CBIS_DDSM.csv'
            df = pd.read_csv(path_csv)

            list_names_img = list(df['file_name_complete'])
            list_names_mask = list(df['file_name_mask'])

            list_img = [path_img + f for f in list_names_img]
            list_mask = [path_mask + f for f in list_names_mask]

        else:
            print("Invalid enter.")

    # Dimensione patches: pw x pw
    pw = 256

    ''' Padding and re-save images in Split_Datasets or Preprocessing '''
    print('doing padding...')
    padding(list_img, list_mask, pw)

    ''' Create and save patches '''
    save_patches(list_img, list_mask, main_path_out, pw, dataset_name)
