import os
import pandas as pd
import cv2
import numpy as np


def reduce_patches(img_path, mask_path, df):
    micro = []
    for filename in list(df['File Name'][df['File Name'].notna()].astype(int)):
        for file in os.listdir(img_path):
            if file.split('_')[0] == str(filename):
                mask = cv2.imread(mask_path + file, cv2.COLOR_BGR2GRAY)
                if np.sum(mask) > 0:
                    micro.append(file)

    for file in os.listdir(img_path):
        if file not in micro:  # list_all:
            os.remove(img_path + file)
            os.remove(mask_path + file)


if __name__ == '__main__':
    train_img_path = 'INbreast/Patches_dataset/train_val_img/'
    train_mask_path = 'INbreast/Patches_dataset/train_val_mask/'

    df_INbreast = pd.read_csv('../../Datasets/INbreast/csv/INbreast_table_noClusters.csv', delimiter=';',
                              encoding="ISO-8859-1")
    df_INbreast['Micros'] = df_INbreast['Micros'].replace('X', 1)
    df_INbreast['Micros'] = df_INbreast['Micros'].replace(np.nan, 0)
    df_INbreast = df_INbreast[df_INbreast['Micros'] == 1]

    reduce_patches(train_img_path, train_mask_path, df_INbreast)
