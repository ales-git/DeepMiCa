from torchmetrics import ConfusionMatrix, AUROC, ROC
from torchmetrics.classification import MulticlassAccuracy as Accuracy
import matplotlib.pyplot as plt
import shap
import torch
import numpy as np
from medcam import medcam
import argparse
import sys
import os

sys.path.append('../_03_FeatureExtraction')
import CustomDataset as dat
import Resnet18
import Vgg16


def testing(dim, batch_size, test_path, test_txt_path, path_checkp):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # data loader
    testdata = dat.CustomDataset(test_path, test_txt_path, dim, extract_name=True)
    testloader = torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=False)
    testloader_shap = torch.utils.data.DataLoader(testdata, batch_size=len(testdata), shuffle=False)

    # load model
    checkpoint = torch.load(path_checkp, map_location=torch.device(device))
    net = checkpoint['model']
    best_epoch = checkpoint['epoch']
    print(net)
    net.load_state_dict(checkpoint['model_state_dict'])

    net = medcam.inject(net, output_dir='attention_maps', backend='gcam', layer='auto',
                        save_maps=True)

    list_pred = []
    list_probs = []
    list_names = []
    list_labels = []

    net.eval()
    for i, data in enumerate(testloader, 0):
        with torch.no_grad():
            inputs, labels, names = data
            inputs, labels = inputs.to(device), labels.to(device)

            print(i)
            print(names)

            outputs = net(inputs)
            outputs = torch.nn.Sigmoid()(outputs.data)
            probs = outputs.clone()
            predicted = torch.round(outputs)

            list_pred.append(predicted)
            list_probs.append(probs)
            list_names.append(names[0])
            list_labels.append(labels.type(torch.int))

    outputs = torch.cat(list_pred).detach().cpu()
    probs = torch.cat(list_probs).detach().cpu()
    names = list_names
    labels = torch.cat(list_labels).detach().cpu()

    shapValues(testloader_shap, path_checkp)

    return outputs, probs, names, labels, best_epoch


def shapValues(testloader, path_checkp):
    checkpoint = torch.load(path_checkp, map_location=torch.device("cpu"))
    net = checkpoint['model']
    net.load_state_dict(checkpoint['model_state_dict'])

    images, labels, names = next(iter(testloader))

    background = images[:100]  # number of background images
    e = shap.DeepExplainer(net, background)

    n_test_images = 1
    test_images = images[0:0 + n_test_images]
    test_labels = np.array(labels[0:0 + n_test_images])

    shap_values = e.shap_values(test_images)

    outputs = net(test_images)
    outputs = torch.nn.Sigmoid()(outputs.data)
    probs = outputs.clone()
    predicted = torch.round(outputs)

    test_images = test_images[:, 0:1, :, :]
    shap_values = [shap_values[:, 0:1, :, :]]

    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

    image_plot(shap_numpy, test_numpy, test_labels, predicted, show=True)


def image_plot(shap_values, pixel_values, labels=None, preds=None, names=None, width=20, aspect=0.2, hspace=0.2,
               labelpad=None, show=True, fig_size=None):
    """ Plots SHAP values for image inputs.
    Parameters
    ----------
    shap_values : [numpy.array]
        List of arrays of SHAP values. Each array has the shap (# samples x width x height x channels), and the
        length of the list is equal to the number of model outputs that are being explained.
    pixel_values : numpy.array
        Matrix of pixel values (# samples x width x height x channels) for each image. It should be the same
        shape as each array in the shap_values list of arrays.
    labels : list
        List of names for each of the model outputs that are being explained. This list should be the same length
        as the shap_values list.
    width : float
        The width of the produced matplotlib plot.
    labelpad : float
        How much padding to use around the model output labels.
    show : bool
        Whether matplotlib.pyplot.show() is called before returning. Setting this to False allows the plot
        to be customized further after it has been created.
    """

    multi_output = False
    if type(shap_values) != list:
        multi_output = False
        shap_values = [shap_values]

    # make sure labels
    if labels is not None:
        assert labels.shape[0] == shap_values[0].shape[0], "Labels must have same row count as shap_values arrays!"
        if multi_output:
            assert labels.shape[1] == len(shap_values), "Labels must have a column for each output in shap_values!"
        else:
            assert len(labels.shape) == 1, "Labels must be a vector for single output shap_values."

    label_kwargs = {} if labelpad is None else {'pad': labelpad}

    # plot our explanations
    x = pixel_values
    if fig_size is None:
        fig_size = np.array([3 * (len(shap_values) + 1), 2.5 * (x.shape[0] + 1)])
        if fig_size[0] > width:
            fig_size *= width / fig_size[0]
    fig, axes = plt.subplots(nrows=x.shape[0], ncols=len(shap_values) + 1, figsize=fig_size)
    if len(axes.shape) == 1:
        axes = axes.reshape(1, axes.size)
    for row in range(x.shape[0]):
        x_curr = x[row].copy()

        # make sure
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 1:
            x_curr = x_curr.reshape(x_curr.shape[:2])
        if x_curr.max() > 1:
            x_curr /= 255.

        # get a grayscale version of the image
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 3:
            x_curr_gray = (
                        0.2989 * x_curr[:, :, 0] + 0.5870 * x_curr[:, :, 1] + 0.1140 * x_curr[:, :, 2])  # rgb to gray
        else:
            x_curr_gray = x_curr

        axes[row, 0].imshow(x_curr, cmap=plt.get_cmap('gray'))
        axes[row, 0].axis('off')
        if len(shap_values[0][row].shape) == 2:
            abs_vals = np.stack([np.abs(shap_values[i]) for i in range(len(shap_values))], 0).flatten()
        else:
            abs_vals = np.stack([np.abs(shap_values[i].sum(-1)) for i in range(len(shap_values))], 0).flatten()
        max_val = np.nanpercentile(abs_vals, 99.9)

        for i in range(len(shap_values)):
            sv = shap_values[i][row] if len(shap_values[i][row].shape) == 2 else shap_values[i][row].sum(-1)
            axes[row, i + 1].imshow(x_curr_gray, cmap=plt.get_cmap('gray'), alpha=0.15,
                                    extent=(-1, sv.shape[1], sv.shape[0], -1))
            im = axes[row, i + 1].imshow(sv, cmap=shap.plots.colors.red_transparent_blue, vmin=-max_val, vmax=max_val)
            axes[row, i + 1].axis('off')
    if hspace == 'auto':
        fig.tight_layout()
    else:
        fig.subplots_adjust(hspace=hspace)
    cb = fig.colorbar(im, ax=np.ravel(axes).tolist(), label="SHAP value", orientation="horizontal",
                      aspect=fig_size[0] / aspect, anchor=(0.5, 2))
    cb.outline.set_visible(False)
    plt.savefig('shap_ex.jpg', dpi=1200)
    if show:
        plt.show()


def metrics(outputs, targets, probs, best_epoch, exp_name):
    # Accuracy
    compute_Accuracy = Accuracy(num_classes=2)
    Accuracy_tot = compute_Accuracy(outputs.squeeze(1), targets)

    # ConfusionMatrix
    compute_CM = ConfusionMatrix(task='binary', num_classes=2)
    ConfusionMatrix_tot = compute_CM(outputs.squeeze(1), targets)
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

    # F1
    if TP.item() == 0:
        F1 = torch.tensor(0)
    else:
        F1 = (2 * precision * recall) / (precision + recall)

    # AUROC
    compute_AUROC = AUROC(task='binary')
    AUROC_tot = compute_AUROC(probs.view(-1), targets.view(-1))

    # File with test results
    f = open(exp_name + ".txt", "w")
    f.write('** Metrics computed on CBIS-DDSM testset **\n\n')
    f.write('Experiment: ' + exp_name + '\n')
    f.write('\nBest epoch: ' + str(best_epoch) + '\n')

    results = ['\n\nAccuracy: ' + str(Accuracy_tot.item()),

               '\n\n ConfusionMatrix: ',
               '\n' + str([TN.item(), FP.item()]),
               '\n' + str([FN.item(), TP.item()]),

               '\n\n AUROC: ' + str(AUROC_tot.item()),
               '\n\n Precision: ' + str(precision.item()),
               '\n\n Recall: ' + str(recall.item()),
               '\n\n Specificity: ' + str(specificity.item()),
               '\n\n F1: ' + str(F1.item())
               ]

    f.writelines(results)
    f.close()


if __name__ == "__main__":

    # set up the parser
    parser = argparse.ArgumentParser(description='micro ROI classification')
    parser.add_argument('--exp_name', type=str, default='classification_experiment_FT_Resnet18', help='experiment name')
    parser.add_argument('--img_dim', type=int, default=224, help='input image dimension')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--n_img_shap', type=int, default=0, help='index of the image to be explained with deepSHAP')
    parser.add_argument('--test_path', type=str, default='./_02_SplitData/CBIS_DDSM/test_img/',
                        help='path to test images')
    parser.add_argument('--test_txt_path', type=str, default='../_02_SplitData/CBIS_DDSM/txt/test.txt',
                        help='path to test txt file')
    args = parser.parse_args()

    if 'FT' in args.exp_name:
        path_checkp = '../_04_FineTuning/Checkpoints/' + args.exp_name + '.pt'
    else:
        path_checkp = '../_03_FeatureExtraction/Checkpoints/' + args.exp_name + '.pt'

    outputs, probs, names, labels, best_epoch = testing(args.img_dim, args.batch_size, args.test_path,
                                                        args.test_txt_path, path_checkp)
    metrics(outputs, labels, probs, best_epoch, args.exp_name)
    print('Done!')
