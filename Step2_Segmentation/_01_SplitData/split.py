import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import os
import numpy as np
from sklearn.model_selection import GroupKFold


def create_files_labels(train, val, test, name, path_towrite, path_files, nfold):
    X_train = train[name]
    X_val = val[name]
    X_test = test[name]

    path_train = path_towrite + 'train_val_img/'
    path_val = path_towrite + 'train_val_img/'
    path_test = path_towrite + 'test_img/'

    path_train_mask = path_towrite + 'train_val_mask/'
    path_val_mask = path_towrite + 'train_val_mask/'
    path_test_mask = path_towrite + 'test_mask/'

    train_list = [path_train + str(s) + '.png ' + path_train_mask + str(s) + '.png' for s in X_train]
    val_list = [path_val + str(s) + '.png ' + path_val_mask + str(s) + '.png' for s in X_val]
    test_list = [path_test + str(s) + '.png ' + path_test_mask + str(s) + '.png' for s in X_test]

    with open(path_files + 'train' + str(nfold) + '.txt', 'w') as f:
        for item in train_list:
            if str(item).startswith(path_train):
                f.write("%s\n" % item)

    with open(path_files + 'validation' + str(nfold) + '.txt', 'w') as f:
        for item in val_list:
            if str(item).startswith(path_val):
                f.write("%s\n" % item)

    with open(path_files + 'test.txt', 'w') as f:
        for item in test_list:
            if str(item).startswith(path_test):
                f.write("%s\n" % item)


def split_dataset(X, id, name, path_out):
    X['Patient ID'] = X['Patient ID'].astype('int')
    train_inds_temp, test_inds = next(
        GroupShuffleSplit(train_size=0.90, n_splits=2, random_state=0).split(X, groups=X[[id]]))
    train_temp = X.iloc[train_inds_temp]
    test = X.iloc[test_inds]

    # kfold grouping by patient
    path_files = 'INbreast/txt_cv/'
    if os.path.exists(path_files) == False:
        os.makedirs(path_files)
    gkf_cv = GroupKFold(n_splits=10)
    for split, (idx_train, idx_test) in enumerate(gkf_cv.split(train_temp, groups=train_temp[[id]])):
        print(f'SPLIT {split + 1}')
        print(f'TRAIN INDEXES: {idx_train}, TEST_INDEXES: {idx_test}\n')
        train_cv = train_temp.iloc[idx_train]
        val_cv = train_temp.iloc[idx_test]
        # Create files with training/validation/test paths and labels
        path_towrite = '../_01_SplitData/INbreast/'

        create_files_labels(train_cv, val_cv, test, name, path_towrite, path_files, nfold=split + 1)


if __name__ == '__main__':
    path_img_in = '../../Step1_Preprocessing/INbreast/AllPng/'
    path_mask_in = '../../Step1_Preprocessing/INbreast/Masks/'
    path_out = 'INbreast/'

    # Read .csv files
    df_INbreast = pd.read_csv('../../Datasets/INbreast/csv/INbreast_table_noClusters.csv', delimiter=';',
                              encoding="ISO-8859-1")
    df_INbreast['Micros'] = df_INbreast['Micros'].replace('X', 1)
    df_INbreast['Micros'] = df_INbreast['Micros'].replace(np.nan, 0)

    X = df_INbreast[df_INbreast['Micros'] == 1]
    X['File Name'] = X['File Name'].astype(int)
    id = 'Patient ID'
    name = 'File Name'

    split_dataset(X, id, name, path_out)
