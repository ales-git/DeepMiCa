import sys
import torch
import os
sys.path.append('../_03_Train')
import SegmentationDataset as dat
from Losses import topk_CE
import cv2
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import MulticlassAccuracy as Accuracy
from torchmetrics.classification import MulticlassJaccardIndex as IoU
from patchify import unpatchify
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import numpy as np


def AND_masks(pred, mask_orig, mask_name, path_predictions):
    mask_orig = (mask_orig / 255).astype('uint8')
    pred = pred.astype('uint8')
    new_pred = cv2.bitwise_and(mask_orig, pred)
    cv2.imwrite(path_predictions + '/' + mask_name, new_pred * 255)


def test_model(dataset_name, inputs_test, targets_test):
    # dataset
    testdata = dat.SegmentationDataSet(inputs_test, targets_test, test=True)

    testloader = torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=False)

    list_images = []
    list_names = []
    list_masks = []

    for (img, mask, names) in testdata:
        list_images.append(img)
        list_masks.append(mask)
        list_names.append(names)
    images = torch.cat(list_images)
    targets = torch.cat(list_masks)

    net.eval()

    list_probs = []
    list_outputs = []

    for i, (x, y, names) in enumerate(testloader, 0):
        with torch.no_grad():
            img, targ = x.squeeze(1).to(device), y.squeeze(1).to(device)
            outputs = net(img)
            outputs = torch.nn.Sigmoid()(outputs)
            if dataset_name == 1:
                probs = outputs.clone()
            else:
                probs = None

            outputs[outputs >= 0.5] = 1
            outputs[outputs < 0.5] = 0

            list_outputs.append(outputs)
            if dataset_name == 1:
                list_probs.append(probs)

    outputs = torch.cat(list_outputs).detach().cpu()

    if dataset_name == 1 or dataset_name == 3:
        probs = torch.cat(list_probs).detach().cpu()

    return images, targets, probs, outputs, list_names


def metrics(outputs, targets):
    # IoU
    compute_IoU = IoU(num_classes=2)
    IoU_mean = compute_IoU(outputs, targets)

    # Accuracy
    compute_Accuracy = Accuracy(num_classes=2)
    Accuracy_mean = compute_Accuracy(outputs, targets)

    # ConfusionMatrix
    compute_CM = ConfusionMatrix(task='binary', num_classes=2)
    ConfusionMatrix_tot = compute_CM(outputs, targets)
    TN = ConfusionMatrix_tot[0][0]
    FP = ConfusionMatrix_tot[0][1]
    FN = ConfusionMatrix_tot[1][0]
    TP = ConfusionMatrix_tot[1][1]

    # Precision
    precision = TP / (TP + FP)

    # Recall
    recall = TP / (TP + FN)

    # Specificity
    specificity = TN / (TN + FP)

    return IoU_mean.item(), Accuracy_mean.item(), TN.item(), FP.item(), FN.item(), TP.item(), precision.item(), recall.item(), specificity.item()


def reconstruct_images(path_predictions, patch_dim, outputs, probs, list_names, dataset_name, path_mask_orig,
                       path_img_orig, mask_name, image_name, filename, best_epoch):
    i = 0  # counter patches
    if dataset_name == 1:
        list_patients = [patch.split('/')[-1].split('_')[0] for patch in list_names]  # get patients
    elif dataset_name == 2:
        list_patients = ['_'.join(patch.split('/')[-1].split('_')[0:3]) for patch in list_names]  # get patients
    else:
        list_patients = [patch.split('/')[-1].split('_')[0] for patch in list_names]  # get patients [0:2]

    list_unique_patients = list(dict.fromkeys(list_patients))  # get unique patients

    if dataset_name == 1 or dataset_name == 3:
        """ Metrics """
        f = open(filename + ".txt", "w")
        f.write('** Metrics computed on INBreast testset **\n\n')
        f.write('Experiment: ' + exp_name + '\n')
        f.write('\nBest epoch: ' + best_epoch + '\n')

        list_IoU_mean, list_Accuracy_mean = [], []
        list_TN, list_FP, list_FN, list_TP = [], [], [], []
        list_AUROC_tot, list_precision, list_recall, list_specificity, list_auc_pr = [], [], [], [], []
        list_all_labels, list_all_probs = [], []

    print(len(list_unique_patients))
    for pat in list_unique_patients:
        print(pat)
        num_patches = list_patients.count(pat)
        outputs_pat = outputs[i:i + num_patches, :, :, :]  # all patches
        if dataset_name == 1:
            probs_pat = probs[i:i + num_patches, :, :, :]  # all patches probs
            ''' We read the original mask to obtain the dim to reconstruct the prediction from the patches.
            It would be the same to read the original image, but we read the mask since we need it later for the metrics.'''
            mask_orig = cv2.imread(path_mask_orig + pat + '.png', cv2.COLOR_BGR2GRAY)
            orig = torch.from_numpy(mask_orig / 255)[:, :]

        elif dataset_name == 2:
            mask_orig = cv2.imread(path_mask_orig + mask_name, cv2.COLOR_BGR2GRAY)
            orig = torch.from_numpy(mask_orig / 255)[:, :]

        # Get dimension of original image and reshape output
        h = int(orig.shape[0] / patch_dim)
        w = int(orig.shape[1] / patch_dim)
        reshaped_outputs_pat = torch.reshape(outputs_pat, (h, w, patch_dim, patch_dim))

        if dataset_name == 1:
            reshaped_probs_pat = torch.reshape(probs_pat, (h, w, patch_dim, patch_dim))
            reconstructed_image = unpatchify(reshaped_outputs_pat.type(torch.int8).numpy(), orig.shape)
            reconstructed_probs = unpatchify(reshaped_probs_pat.numpy(), orig.shape)
            if dataset_name == 1:
                list_all_labels.append(list(orig.type(torch.int8).view(-1).numpy()))
                list_all_probs.append(list(reconstructed_probs.flatten()))
            else:
                # list_all_probs.append(list(reconstructed_probs.flatten()))
                list_all_labels = [0] * len(orig.type(torch.int8).view(-1).numpy())

            '''Metrics'''
            IoU_mean, Accuracy_mean, TN, FP, FN, TP, precision, recall, specificity = metrics(
                torch.from_numpy(reconstructed_image), orig.type(torch.int8))

            # IoU and Accuracy
            list_IoU_mean.append(IoU_mean),
            list_Accuracy_mean.append(Accuracy_mean)

            # Confusion Matrix
            list_TN.append(TN), list_FP.append(FP), list_FN.append(FN), list_TP.append(TP)

            # precision, recall, specificity
            list_precision.append(precision),
            list_recall.append(recall),
            list_specificity.append(specificity)

            ''' Save pred masks '''
            cv2.imwrite(path_predictions + '/' + pat + ".png", reconstructed_image * 255)
            del IoU_mean, Accuracy_mean, TN, FP, FN, TP
            del precision, recall, specificity
            del reconstructed_image
            del orig
            del outputs_pat, reshaped_outputs_pat
            print()

        elif dataset_name == 2:
            reconstructed_image = unpatchify(reshaped_outputs_pat.type(torch.int8).numpy(),
                                             (orig.shape[0], orig.shape[1]))

            ''' Logical AND and Save pred masks '''
            # Split masks if the original image had more than one and save them separately
            AND_masks(reconstructed_image, mask_orig, mask_name, path_predictions)

        i = i + num_patches

    if dataset_name == 1:
        lists = [list_IoU_mean, list_Accuracy_mean,
                 list_TN, list_FP, list_FN, list_TP,
                 list_precision, list_recall,
                 list_specificity]

        l_mean = [sum(l) / len(l) for l in lists]

        # ROC
        y_true = np.hstack(list_all_labels)
        y_prob = np.hstack(list_all_probs)
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc_roc = auc(fpr, tpr)

        # Precision recall curve
        y_true = np.hstack(list_all_labels)
        y_prob = np.hstack(list_all_probs)
        prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
        auc_pr = auc(rec, prec)

        results = [
            '\nIoU_mean: ' + str(l_mean[0]),
            '\nAccuracy_mean: ' + str(l_mean[1]),
            '\n\n ConfusionMatrix: ',
            '\n' + str([sum(list_TN), sum(list_FP)]),
            '\n' + str([sum(list_FN), sum(list_TP)]),
            '\n\n AUROC: ' + str(auc_roc),
            '\n\n AUC_PRECISION_RECALL: ' + str(auc_pr),
            '\n\n Precision: ' + str(l_mean[6]),
            '\n\n Recall: ' + str(l_mean[7]),
            '\n\n Specificity: ' + str(l_mean[8])
        ]
        f.writelines(results)
        f.close()


#######################################################################################################
if __name__ == '__main__':

    # Choose the dataset to use
    dataset_name = 0
    while dataset_name not in [1, 2]:

        dataset_name = int(
            input("Choose the dataset to use (1 = INbreast, 2 = CBIS-DDSM):\n"))

        if dataset_name == 1:
            main_path = '../_02_Patches/INbreast/'
            # Path patches
            path_patches = os.path.join(main_path, 'Patches_dataset/')
            path_test_img = os.path.join(path_patches, 'test_img/')
            path_test_mask = os.path.join(path_patches, 'test_mask/')
            # Path complete
            path_img_orig = '../../Step1_Preprocessing/INbreast/AllPng/'
            path_mask_orig = '../../Step1_Preprocessing/INbreast/Masks/'

            # test img + mask
            inputs_test = [path_test_img + str(x) for x in os.listdir(path_test_img)]
            targets_test = [path_test_mask + str(x) for x in os.listdir(path_test_mask)]


        elif dataset_name == 2:
            main_path = '../_02_Patches/CBIS_DDSM/'
            # Path patches
            path_patches = main_path + 'Patches_dataset/'
            path_test_img = path_patches + 'AllPng/'
            path_test_mask = path_patches + 'Masks/'
            # Path complete
            path_mask_orig = '../../Step1_Preprocessing/CBIS_DDSM/Masks/'
            path_img_orig = '../../Step1_Preprocessing/CBIS_DDSM/AllPng/'
            path_csv = '../../Datasets/CBIS_DDSM/csv/CBIS_DDSM.csv'
            # Path csv
            df = pd.read_csv(path_csv)
            name_complete_csv = 'file_name_complete'
            name_mask_csv = 'file_name_mask'

        else:
            print("Invalid enter.")

    ################################################################################################
    # Experiment name
    exp_name = 'insert_segmentation_exp_name_here' #TODO: insert segmentation experiment name here
    if exp_name == 'insert_segmentation_exp_name_here':
        print("Insert a valid experiment name.")
        exit()
    # batch_size
    batch_size = 1
    # Loss
    criterion = topk_CE()
    # Patch width
    pw = 256
    ################################################################################################

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    main_path_weights = '../Checkpoints/'
    if os.path.exists(main_path_weights) == False:
        os.makedirs(main_path_weights)
    path_weights = main_path_weights + exp_name + '.pt'

    path_predictions = os.path.join('Predictions', main_path.split('/')[2])
    if os.path.exists(path_predictions) == False:
        os.makedirs(path_predictions)

    path_metrics = 'Metrics_test/'
    if os.path.exists(path_metrics) == False:
        os.makedirs(path_metrics)

    # Load model and weights
    checkpoint = torch.load(path_weights, map_location=torch.device(device))
    net = checkpoint['model']
    best_epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['model_state_dict'])

    ''' Test model '''
    if dataset_name == 1:
        images, targets, probs, outputs, list_names = test_model(dataset_name, inputs_test, targets_test)
        mask_name = None
        image_name = None

        del net
        print('finished test.')

        ''' Reconstruct and save predictions '''
        reconstruct_images(path_predictions, pw, outputs, probs, list_names, dataset_name, path_mask_orig,
                           path_img_orig,
                           mask_name, image_name, filename=path_metrics + exp_name, best_epoch=str(best_epoch))

    elif dataset_name == 2:
        for (image_name, mask_name) in zip(list(df[name_complete_csv]), list(df[name_mask_csv])):
            inputs_test = [path_test_img + str(x) for x in os.listdir(path_test_img) if
                           x.startswith(image_name.split('.')[0])]
            targets_test = [path_test_mask + str(x) for x in os.listdir(path_test_mask) if
                            x.startswith(mask_name.split('.')[0])]

            print("##############################")
            print("Testing:")
            print(image_name)
            print(mask_name)

            images, targets, probs, outputs, list_names = test_model(dataset_name, inputs_test, targets_test)

            ''' Reconstruct and save predictions '''
            reconstruct_images(path_predictions, pw, outputs, probs, list_names, dataset_name, path_mask_orig,
                               path_img_orig,
                               mask_name, image_name, filename=path_metrics + exp_name, best_epoch=str(best_epoch))

            print("##############################")
            print('\n')
