import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def ROI_coord(mask):
    index_x = [i for i in range(mask.shape[1]) if np.sum(mask[:, i]) > 0]
    x1 = index_x[0]
    x2 = index_x[-1]

    index_y = [i for i in range(mask.shape[0]) if np.sum(mask[i, :]) > 0]
    y1 = index_y[0]
    y2 = index_y[-1]

    return [x1, x2, y1, y2]


def ROI_center(mask):
    roi_coord = ROI_coord(mask)
    x_center = int(roi_coord[0] + (roi_coord[1] - roi_coord[0]) / 2)
    y_center = int(roi_coord[2] + (roi_coord[3] - roi_coord[2]) / 2)
    return [x_center, y_center]


def image_center(img):
    x_center = int(img.shape[1] / 2)
    y_center = int(img.shape[0] / 2)
    return [x_center, y_center]


def find_mammography_border(cut_image, dist):
    index = np.where(cut_image == 0)[0].tolist()
    if len(index) > 0:
        dist.append(index[0])
    else:
        dist.append(0)
    return dist


def color_black(img, mask, binary, ROI_center_coord, side):
    dist = []
    if side == 'right':
        for i in range(binary.shape[0]):
            cut_image = binary[i, ROI_center_coord[0]:binary.shape[1]]
            dist = find_mammography_border(cut_image, dist)

        img_tocolor = round((binary.shape[1] - ROI_center_coord[0] - max(dist)) * 0.9)
        img[:, (img.shape[1] - img_tocolor):img.shape[1]] = 0
        binary[:, (binary.shape[1] - img_tocolor):binary.shape[1]] = 0

        # plt.figure(0)
        # plt.imshow(img, cmap='gray')
        # plt.show()

    if side == 'down':
        for i in range(binary.shape[1]):
            cut_image = binary[ROI_center_coord[1]:binary.shape[0], i]
            dist = find_mammography_border(cut_image, dist)

        img_tocolor = round((binary.shape[0] - ROI_center_coord[1] - max(dist)) * 0.9)
        img[(img.shape[0] - img_tocolor):img.shape[0], :] = 0
        binary[(binary.shape[0] - img_tocolor):binary.shape[0], :] = 0

        # plt.figure(1)
        # plt.imshow(img, cmap='gray')
        # plt.show()

    if side == 'up':

        # flip mask and recompute center roi
        if mask is not None:
            mask = np.flipud(mask)
            ROI_center_coord = ROI_center(mask)

        binary = np.flipud(binary)
        for i in range(binary.shape[1]):
            cut_image = binary[ROI_center_coord[1]:binary.shape[0], i]
            dist = find_mammography_border(cut_image, dist)

        img_tocolor = round((binary.shape[0] - ROI_center_coord[1] - max(dist)) * 0.9)

        # img
        img = np.flipud(img)
        img[(img.shape[0] - img_tocolor):img.shape[0], :] = 0
        img = np.flipud(img)
        # mask
        if mask is not None:
            mask = np.flipud(mask)
        # binary
        binary[(img.shape[0] - img_tocolor):img.shape[0], :] = 0
        binary = np.flipud(binary)

        # plt.figure(2)
        # plt.imshow(img, cmap='gray')
        # plt.show()

    if side == 'left':
        # on the left side there should be the breast, but sometimes the scannner can crate a white line
        # so we take only the first 1% of the image from the left and we check if at least 80% of the pixels
        # are white; if so, we delete this part of the image from both the complete one and the corresponding mask.

        cut_image = img[:, 0:round(binary.shape[1] * 0.001)]
        if (len(np.where(cut_image > 0.9)[0]) > 0.1 * (cut_image.size)) or (
                len(np.where(cut_image < 0.1)[0]) > 0.1 * (cut_image.size)):
            img = img[:, round(binary.shape[1] * 0.01): img.shape[1]]
            if mask is not None:
                mask = mask[:, round(binary.shape[1] * 0.01): mask.shape[1]]
            binary = binary[:, round(binary.shape[1] * 0.01): binary.shape[1]]

    return img, mask, binary


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
            # Then dilate
            edited_mask = cv2.morphologyEx(edited_mask, cv2.MORPH_DILATE, kernel)

        elif operation == "close":
            edited_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            # Then dilate
            edited_mask = cv2.morphologyEx(edited_mask, cv2.MORPH_DILATE, kernel)

        elif operation == "dilation":
            edited_mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        elif operation == "erosion":
            edited_mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)

    except Exception as e:
        # logger.error(f'Unable to editMask!\n{e}')
        print((f"Unable to get editMask!\n{e}"))

    return edited_mask


def sortContoursByArea(contours, reverse=True):
    """
    This function takes in list of contours, sorts them based
    on contour area, computes the bounding rectangle for each
    contour, and outputs the sorted contours and their
    corresponding bounding rectangles.
    Parameters
    ----------
    contours : {list}
        The list of contours to sort.
    Returns
    -------
    sorted_contours : {list}
        The list of contours sorted by contour area in descending
        order.
    bounding_boxes : {list}
        The list of bounding boxes ordered corresponding to the
        contours in `sorted_contours`.
    """

    try:
        # Sort contours based on contour area.
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=reverse)

        # Construct the list of corresponding bounding boxes.
        bounding_boxes = [cv2.boundingRect(c) for c in sorted_contours]

    except Exception as e:
        print((f"Unable to get sortContourByArea!\n{e}"))

    return sorted_contours, bounding_boxes


def xLargestBlobs(mask, top_x=None, reverse=True):
    """
    This function finds contours in the given image and
    keeps only the top X largest ones.
    Parameters
    ----------
    mask : {numpy.ndarray, dtype=np.uint8}
        The mask to get the top X largest blobs.
    top_x : {int}
        The top X contours to keep based on contour area
        ranked in decesnding order.
    Returns
    -------
    n_contours : {int}
        The number of contours found in the given `mask`.
    X_largest_blobs : {numpy.ndarray}
        The corresponding mask of the image containing only
        the top X largest contours in white.
    """
    try:
        # Find all contours from binarised image.
        contours, hierarchy = cv2.findContours(
            image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE
        )

        n_contours = len(contours)

        # Only get largest blob if there is at least 1 contour.
        if n_contours > 0:

            if n_contours < top_x or top_x == None:
                top_x = n_contours

            # Sort contours based on contour area.
            sorted_contours, bounding_boxes = sortContoursByArea(
                contours=contours, reverse=reverse
            )

            # Get the top X largest contours.
            X_largest_contours = sorted_contours[0:top_x]

            # Create black canvas to draw contours on.
            to_draw_on = np.zeros(mask.shape, np.uint8)

            # Draw contours in X_largest_contours.
            X_largest_blobs = cv2.drawContours(
                image=to_draw_on,  # Draw the contours on `to_draw_on`.
                contours=X_largest_contours,  # List of contours to draw.
                contourIdx=-1,  # Draw all contours in `contours`.
                color=1,  # Draw the contours in white.
                thickness=-1,  # Thickness of the contour lines.
            )

    except Exception as e:
        print((f"Unable to get xLargestBlobs!\n{e}"))

    return n_contours, X_largest_blobs


def applyMask(img, mask):
    """
    This function applies a mask to a given image. White
    areas of the mask are kept, while black areas are
    removed.
    Parameters
    ----------
    img : {numpy.ndarray}
        The image to mask.
    mask : {numpy.ndarray, dtype=np.uint8}
        The mask to apply.
    Returns
    -------
    masked_img: {numpy.ndarray}
        The masked image.
    """

    try:
        masked_img = img.copy()
        masked_img[mask == 0] = 0

    except Exception as e:
        print((f"Unable to get applyMask!\n{e}"))

    return masked_img


def crop_image_outside(img, img_mask, tol=0):
    # plt.figure(0)
    # plt.imshow(img, cmap='gray')
    # plt.show()

    mask = img > tol
    if img.ndim == 3:
        mask = mask.all(2)
    m, n = mask.shape
    print('shape: ', (m, n))
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
    print('col-row: ', (col_start, col_end, row_start, row_end))
    if img_mask is not None:
        return img[row_start:row_end, col_start:col_end], img_mask[row_start:row_end, col_start:col_end]
    else:
        return img[row_start:row_end, col_start:col_end], img_mask


def Clahe(img):
    img = img * 255
    img = img.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl_img = clahe.apply(img)

    return cl_img


def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)


def save_images(img, mask, image_name, mask_name, path_img_save, path_mask_save):
    print(image_name)
    print('Saving preprocessed image...')
    if os.path.exists(path_img_save) == False:
        os.makedirs(path_img_save)
    cv2.imwrite(os.path.join(path_img_save, image_name), img)
    if mask is not None:
        if os.path.exists(path_mask_save) == False:
            os.makedirs(path_mask_save)
        cv2.imwrite(path_mask_save + mask_name, mask)
    print('Done!')

    # plt.figure(2)
    # plt.imshow(img,cmap='gray')
    # plt.figure(3)
    # plt.imshow(cl_img, cmap='gray')


def preprocessing(df, path_img, path_mask, path_img_out, path_mask_out, name_complete_csv, name_mask_csv, dataset_name,
                  save):
    for (image_name, mask_name) in zip(list(df[name_complete_csv]), list(df[name_mask_csv])):

        if dataset_name == 1:
            image_name = str(image_name) + '.png'
            mask_name = str(mask_name) + '.png'
        img = cv2.imread(path_img + image_name, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(path_mask + mask_name, cv2.IMREAD_GRAYSCALE)

        """ Flip if the breast is on the right """
        # If the sum of the pixels values in the right part of the image is > than the left part of the image,
        # the breast is on the right ad it will be flipped to the left.
        left_image_part = np.sum(img[0:img.shape[0], 0:round(img.shape[1] / 2)])
        right_image_part = np.sum(img[0:img.shape[0], round(img.shape[1] / 2):img.shape[1]])

        if (right_image_part > left_image_part):
            img = np.fliplr(img)
            mask = np.fliplr(mask)

        """ (x,y) Coordinates of the ROI center"""
        ROI_center_coord = ROI_center(mask)

        """ Fill with black pixels """

        # 1. Binarize image
        img = img / 255
        _, binary = cv2.threshold(img, 0.01, 1, cv2.THRESH_BINARY)

        print(img.shape)
        print(mask.shape)
        # 2. Color right part
        img, mask, binary = color_black(img, mask, binary, ROI_center_coord, side='right')
        # 3. Color bottom part
        img, mask, binary = color_black(img, mask, binary, ROI_center_coord, side='down')
        # 4. Color upper part
        img, mask, binary = color_black(img, mask, binary, ROI_center_coord, side='up')
        # 4. Cut left part
        img, mask, binary = color_black(img, mask, binary, ROI_center_coord, side='left')
        print(img.shape)
        print(mask.shape)

        """ Remove artifacts """
        _, binary = cv2.threshold(img, 0.01, 1, cv2.THRESH_BINARY)
        binary = binary.astype(np.uint8)

        edited_mask = editMask(binary)
        _, xlargest_mask = xLargestBlobs(edited_mask, top_x=1, reverse=True)
        img = applyMask(img, mask=xlargest_mask)

        ''' Crop image background '''
        img, mask = crop_image_outside(img, mask)

        print(img.shape)
        print(mask.shape)

        """ Clahe """
        img = Clahe(img)

        # 7. Save all preprocessed images and masks
        if save == True:
            save_images(img, mask, image_name, mask_name, path_img_out, path_mask_out)


if __name__ == '__main__':

    # Choose the dataset to use
    dataset_name = 0
    while dataset_name not in [1, 2]:

        dataset_name = int(
            input("Choose the dataset to use (1 = INbreast, 2 = CBIS-DDSM):\n"))

        if dataset_name == 1:
            path_csv = '../Datasets/INbreast/csv/INbreast_table_noClusters.csv'
            path_img_in = '../Datasets/INbreast/AllPng/'
            path_mask_in = '../Datasets/INbreast/Masks/'
            path_img_out = 'INbreast/AllPng/'
            path_mask_out = 'INbreast/Masks/'
            name_complete_csv = 'File Name'
            name_mask_csv = 'File Name'
            df_INbreast = pd.read_csv(path_csv, delimiter=';', encoding="ISO-8859-1")
            df_INbreast['Micros'] = df_INbreast['Micros'].replace('X', 1)
            df_INbreast['Micros'] = df_INbreast['Micros'].replace(np.nan, 0)
            micros = df_INbreast['File Name'][df_INbreast['Micros'] == 1].astype(int)
            df = df_INbreast[df_INbreast['Micros'] == 1]
            df['File Name'] = df['File Name'].astype(int)


        elif dataset_name == 2:
            path_csv = '../Datasets/CBIS_DDSM/csv/CBIS_DDSM.csv'
            path_img_in = '../Datasets/CBIS_DDSM/AllPng/'
            path_mask_in = '../Datasets/CBIS_DDSM/Masks/'
            path_img_out = 'CBIS_DDSM/AllPng/'
            path_mask_out = 'CBIS_DDSM/Masks/'
            name_complete_csv = 'file_name_complete'
            name_mask_csv = 'file_name_mask'
            df = pd.read_csv(path_csv)

        else:
            print("Invalid enter.")

    # preprocessing train
    preprocessing(df, path_img_in, path_mask_in, path_img_out, path_mask_out, name_complete_csv, name_mask_csv,
                  dataset_name, save=True)
