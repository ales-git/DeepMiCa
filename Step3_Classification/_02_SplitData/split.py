import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import os
import shutil


def prepare_folder(path_in, path_out):
    if not os.path.exists(path_out + 'train_img/'):
        shutil.copytree(path_in, path_out + 'train_img/')
    else:
        print('Train img folder already exists')

    if not os.path.exists(path_out + 'val_img/'):
        os.mkdir(path_out + 'val_img')

    if not os.path.exists(path_out + 'test_img/'):
        os.mkdir(path_out + 'test_img')

    if not os.path.exists(path_out + 'txt'):
        os.mkdir(path_out + 'txt')


def split_dataset(X, id, name, path_out):
    train_inds_temp, test_inds = next(
        GroupShuffleSplit(train_size=0.90, n_splits=2, random_state=0).split(X, groups=X[[id]]))
    train_temp = X.iloc[train_inds_temp]
    test = X.iloc[test_inds]

    train_inds, val_inds = next(
        GroupShuffleSplit(train_size=0.90, n_splits=2, random_state=0).split(train_temp, groups=train_temp[[id]]))
    train = train_temp.iloc[train_inds]
    val = train_temp.iloc[val_inds]

    print('Len train: ', len(train))
    print('Len val: ', len(val))
    print('Len test: ', len(test))
    print()

    for filename in list(test[name]):
        train_img_list = [f.split('.')[0] + '.png' for f in os.listdir(path_out + 'train_img/') if
                          not f.startswith('.')]

        if filename in train_img_list:
            shutil.move(path_out + 'train_img/' + str(filename), path_out + 'test_img/')

    for filename in list(val[name]):
        train_img_list = [f.split('.')[0] + '.png' for f in os.listdir(path_out + 'train_img/') if
                          not f.startswith('.')]

        if filename in train_img_list:
            shutil.move(path_out + 'train_img/' + str(filename), path_out + 'val_img/')

    return train, val, test


# Create files with training/validation/test paths and labels
def create_files_labels(train, val, test, name, label, path_towrite, path_files):
    X_train, y_train = train[name], train[label]
    X_val, y_val = val[name], val[label]
    X_test, y_test = test[name], test[label]

    path_train = path_towrite + 'train_img/'
    path_val = path_towrite + 'val_img/'
    path_test = path_towrite + 'test_img/'

    train_list = [path_train + str(s) + ' ' + str(s1) for s, s1 in zip(X_train, y_train)]
    val_list = [path_val + str(s) + ' ' + str(s1) for s, s1 in zip(X_val, y_val)]
    test_list = [path_test + str(s) + ' ' + str(s1) for s, s1 in zip(X_test, y_test)]

    with open(path_files + 'train.txt', 'w') as f:
        for item in train_list:
            if str(item).startswith(path_train):
                f.write("%s\n" % item)

    with open(path_files + 'validation.txt', 'w') as f:
        for item in val_list:
            if str(item).startswith(path_val):
                f.write("%s\n" % item)

    with open(path_files + 'test.txt', 'w') as f:
        for item in test_list:
            if str(item).startswith(path_test):
                f.write("%s\n" % item)


if __name__ == '__main__':

    path_in = '../_01_CutROI/CBIS_DDSM/Cropped/'

    X = pd.read_csv('../../Datasets/CBIS_DDSM/csv/CBIS_DDSM.csv')  # CBIS_DDSM
    id = 'patient_id'
    name = 'file_name_mask'
    label = 'pathology'

    path_out = 'CBIS_DDSM/'
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    path_towrite = os.path.join(os.getcwd(), path_out)
    path_files = path_out + 'txt/'

    prepare_folder(path_in, path_out)
    train, val, test = split_dataset(X, id, name, path_out)
    create_files_labels(train, val, test, name, label, path_towrite, path_files)
