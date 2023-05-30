import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
This code creates squares with the minimum size around the microcalcifications,
and saves them as new images.
'''


def ROI_coord(mask):
    index_x = [i for i in range(mask.shape[1]) if np.sum(mask[:, i]) > 0]
    x1 = index_x[0]
    x2 = index_x[-1]

    index_y = [i for i in range(mask.shape[0]) if np.sum(mask[i, :]) > 0]
    y1 = index_y[0]
    y2 = index_y[-1]

    return [x1, x2, y1, y2]


def plot_ROI_rectangle(binaryRGB, x1, x2, y1, y2):
    for j in range(5):
        binaryRGB[y1:y2, x1 + j, 1] = 1
        binaryRGB[y1:y2, x2 - j, 1] = 1
        binaryRGB[y1 + j, x1:x2, 1] = 1
        binaryRGB[y2 - j, x1:x2, 1] = 1

    plt.figure("red rectangle")
    plt.imshow(binaryRGB)
    plt.show()


def cut_green_square_Roi(binaryRGB, x1, x2, y1, y2, side):
    # x delta
    x_delta = side - (x2 - x1)
    if (x_delta / 2 > x1 or x_delta / 2 > (binaryRGB.shape[1] - x2)):
        if (x_delta / 2 > x1):
            x1_fin = 0
            x2_fin = side
        if (x_delta / 2 > (binaryRGB.shape[1] - x2)):
            x1_fin = binaryRGB.shape[1] - side
            x2_fin = binaryRGB.shape[1]

    else:
        if ((x_delta % 2) != 0):
            x1_fin = x1 - round(x_delta / 2)
            x2_fin = x1_fin + side
        else:
            if (x_delta == 0):
                x1_fin = x1
                x2_fin = x2
            else:
                x1_fin = x1 - (x_delta / 2)
                x2_fin = x1_fin + side

    # y delta
    y_delta = side - (y2 - y1)
    if y_delta / 2 > y1 or y_delta / 2 > (binaryRGB.shape[0] - y2):
        if y_delta / 2 > y1:
            y1_fin = 0
            y2_fin = side
        if y_delta / 2 > (binaryRGB.shape[0] - y2):
            y1_fin = binaryRGB.shape[0] - side
            y2_fin = binaryRGB.shape[0]

    else:
        if (y_delta % 2) != 0:
            y1_fin = y1 - round(y_delta / 2)
            y2_fin = y1_fin + side
        else:
            if y_delta == 0:
                y1_fin = y1
                y2_fin = y2
            else:
                y1_fin = y1 - (y_delta / 2)
                y2_fin = y1_fin + side

    x1_fin = int(x1_fin)
    x2_fin = int(x2_fin)
    y1_fin = int(y1_fin)
    y2_fin = int(y2_fin)

    return x1_fin, x2_fin, y1_fin, y2_fin


def cut_blue_square_Roi(binary, x1_red, x2_red, y1_red, y2_red, x1_green, x2_green, y1_green, y2_green, side):
    """ Move the ROI horizontally to exclude black pixels """

    if x2_red > side:
        right = 0
        left = 0

        for r in range(x2_red, x2_green):
            right = right + sum(binary[y1_green:y2_green, r] == 0)  # number of black pixels from x2_red to x2_green
        for l in range(x1_green, x1_red):
            left = left + sum(binary[y1_green:y2_green, l] == 0)  # number of black pixels from x1_green to x1_red

        # If the number of black pixels on the right is greater than the number of black pixels on the left,
        # the squared is moved from right to left
        if right > left:
            dist = []
            for y_k in range(y1_green, y2_green):
                vect = binary[y_k, x2_red:binary.shape[1]]
                index_black = np.where(vect == 0)[0]

                if len(index_black) > 0:
                    index_white = np.where(vect[0:index_black[0]] == 1)[0]
                    dist.append(len(index_white))

            min_dist = min(dist)
            x2_blue = x2_red + min_dist
            x1_blue = x2_blue - side

        # If the number of black pixels on the left is greater than the number of black pixels on the right,
        # the squared is moved from left to right (Since all the images are flipped this should never happen)
        if left >= right:
            x1_blue = x1_green
            x2_blue = x2_green

    else:
        x1_blue = x1_green
        x2_blue = x2_green

    """ Move the ROI vertically to exclude black pixels """

    flag = np.any(binary[x1_blue:x2_blue, y1_green:y2_green] == 0)  # True if there are black pixels in the new square

    if y1_red > side & flag:
        up = 0
        down = 0

        for u in range(y1_green, y1_red):
            up = up + sum(binary[u, x1_blue:x2_blue] == 0)

        for d in range(y2_red, y2_green):
            down = down + sum(binary[d, x1_blue:x2_blue] == 0)

        if down > up:
            dist = []
            for x_k in range(x1_blue, x2_blue):
                vect = binary[y2_red:binary.shape[0], x_k]
                index_black = np.where(vect == 0)[0]

                if len(index_black) > 0:
                    index_white = np.where(vect[0:index_black[0]] == 1)[0]
                    dist.append(len(index_white))

            min_dist = min(dist)
            y2_blue = y2_red + min_dist
            y1_blue = y2_blue - side

        if up > down:
            dist = []
            for x_k in range(x1_blue, x2_blue):
                vect = binary[0:y1_red, x_k]
                index_black = np.where(vect == 0)[0]

                if len(index_black > 0):
                    index_white = np.where(vect[index_black[-1]:len(vect)] == 1)[0]
                    dist.append(len(index_white))

            min_dist = min(dist)
            y1_blue = y1_red - min_dist
            y2_blue = y1_blue + side

        else:
            y1_blue = y1_green
            y2_blue = y2_green
    else:
        y1_blue = y1_green
        y2_blue = y2_green

    return x1_blue, x2_blue, y1_blue, y2_blue


def plot_ROI_green_square(binaryRGB, x1, x2, y1, y2):
    for j in range(5):
        binaryRGB[y1:y2, x1 + j, 2] = 1
        binaryRGB[y1:y2, x2 - j, 2] = 1
        binaryRGB[y1 + j, x1:x2, 2] = 1
        binaryRGB[y2 - j, x1:x2, 2] = 1

    plt.figure("green square")
    plt.imshow(binaryRGB)
    plt.show()


def plot_ROI_blue_square(binaryRGB, x1, x2, y1, y2):
    for j in range(5):
        binaryRGB[y1:y2, x1 + j, 0] = 1
        binaryRGB[y1:y2, x2 - j, 0] = 1
        binaryRGB[y1 + j, x1:x2, 0] = 1
        binaryRGB[y2 - j, x1:x2, 0] = 1

    plt.figure("blue square", figsize=(8, 10))
    plt.imshow(binaryRGB)
    plt.axis('off')
    plt.show()


def editMask(mask, ksize=(23, 23), operation="open"):
    """
    This function edits a given mask (binary image) by performing
    closing then opening morphological operations.
    Parameters
    ----------
    mask : {numpy.ndarray}
        The mask to edit.
    ksize : {tuple}
        Size of the structuring element.
    operation : {str}
        Either "open" or "close", each representing open and close
        morphological operations respectively.
    Returns
    -------
    edited_mask : {numpy.ndarray}
        The mask after performing close and open morphological
        operations.
    """

    try:
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=ksize)

        if operation == "open":
            edited_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        elif operation == "close":
            edited_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Then dilate
        edited_mask = cv2.morphologyEx(edited_mask, cv2.MORPH_DILATE, kernel)

    except Exception as e:
        print((f"Unable to get editMask!\n{e}"))

    return edited_mask


def save_images(cropped_img, path_cropped, mask_name):
    print('Saving cropped image...')
    cropped_img = (cropped_img - cropped_img.min()) / (cropped_img.max() - cropped_img.min())  # normalize
    cv2.imwrite(path_cropped + mask_name, cropped_img * 255)
    print(cropped_img.max())
    print(cropped_img.min())
    print('Done!')


def Clahe(img):
    # img = img*255
    img = img.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl_img = clahe.apply(img)

    return cl_img


def cutRoi(path_img, path_mask, path_cropped, path_csv, name_complete_csv, name_mask_csv, df):

    for (image_name, mask_name) in zip(list(df[name_complete_csv]), list(df[name_mask_csv])):

        print(mask_name)
        img = cv2.imread(path_img + image_name, cv2.IMREAD_GRAYSCALE)
        img = Clahe(img)
        mask = cv2.imread(path_mask + mask_name, cv2.IMREAD_GRAYSCALE)

        img = img / 255
        _, binary = cv2.threshold(img, 0.07, 1, cv2.THRESH_BINARY)
        binary = binary.astype(np.uint8)

        # Open and close operations to adjust borders
        binary = editMask(binary, ksize=(11, 11), operation="open")
        binary = editMask(binary, ksize=(5, 5), operation="close")

        binaryRGB = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB) * 255

        # Trasform Roi as gray
        binaryRGB[:, :, 0] = binaryRGB[:, :, 0] + 224 / 255 * mask
        binaryRGB[:, :, 1] = binaryRGB[:, :, 1] + 224 / 255 * mask
        binaryRGB[:, :, 2] = binaryRGB[:, :, 2] + 224 / 255 * mask
        binaryRGB = binaryRGB.astype(np.uint8)

        # Roi coord
        coord = ROI_coord(mask)
        x1 = coord[0]
        x2 = coord[1]
        y1 = coord[2]
        y2 = coord[3]

        # Plot
        # plot_ROI_rectangle(binaryRGB,x1,x2,y1,y2)

        # Bigger side of the rectangle
        side = max((x2 - x1), (y2 - y1))
        if side < 64:
            side = 64

        if side >= binary.shape[0] or side >= binary.shape[1]:
            print('****************************************')
            print('\nMask too big:')
            # masks_toobig.append(mask_name)

            """ Crop image """
            cropped_img = img[y1:y2, x1:x2]

            """ Save cropped image """
            save_images(cropped_img, path_cropped, mask_name)

            print(mask_name)
            print('****************************************')
            continue

        # Green square
        x1_fin, x2_fin, y1_fin, y2_fin = cut_green_square_Roi(binary, x1, x2, y1, y2, side)

        # plot_ROI_green_square(binaryRGB,x1_fin,x2_fin,y1_fin,y2_fin)

        # Blue square
        flag = np.any(binary[x1_fin:x2_fin, y1_fin:y2_fin] == 0)  # True if there are black pixels in the green square

        if flag:
            x1_fin, x2_fin, y1_fin, y2_fin = cut_blue_square_Roi(binary, x1, x2, y1, y2, x1_fin, x2_fin, y1_fin, y2_fin,
                                                                 side)
            # plot_ROI_blue_square(binaryRGB, x1_fin, x2_fin, y1_fin, y2_fin)

        """ Crop image """
        cropped_img = img[y1_fin:y2_fin, x1_fin:x2_fin]

        """ Save cropped image """
        save_images(cropped_img, path_cropped, mask_name)


if __name__ == '__main__':

    """ Paths """
    path_img = '../../Datasets/CBIS_DDSM/AllPng/'
    path_mask = '../../Datasets/CBIS_DDSM/Masks/'
    name_complete_csv = 'file_name_complete'
    name_mask_csv = 'file_name_mask'
    path_csv = '../../Datasets/CBIS_DDSM/csv/CBIS_DDSM.csv'

    path_cropped = 'CBIS_DDSM/Cropped/'
    if not os.path.exists(path_cropped):
        os.makedirs(path_cropped)

    df = pd.read_csv(path_csv, delimiter=',')

    """ Cut Roi """
    cutRoi(path_img, path_mask, path_cropped, path_csv, name_complete_csv, name_mask_csv, df)
