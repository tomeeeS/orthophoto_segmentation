# Read in a couple of test images and do the following for each: make overlapping patches of the
#   image, get predictions for each patch, reassemble them and plot the predictions with the
#   original image and the ground truth
# Can toggle if we want to use ResUNet-a too or only UNet
# Can toggle if we want to use the GPU, or the CPU (useful if we are currently training on the GPU)

# If running out of memory, try one or more of the following:
#   set WITH_TEST_TIME_AUGMENTATION to False,
#   decrease PREDICTION_SIZE_MULTIPLIER,
#   RUN_ON_CPU to True (to run with RAM and not GPU's memory)

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import tensorflow as tf
from cv2 import cv2
from rasterio.plot import show
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.metrics import MeanIoU

from cfg import *
from data_handling import from_one_hot_to_category_indices, read_elevation, add_channel, \
    add_masked_alpha_channel, segmentation_map_from_mask, normalize
from data_handling import read_one_band, DOWNSCALING_RATE
from patching import change_some_categories
from smooth_tiled_predictions import predict_img_with_smooth_windowing
from train import get_model_unet, unet_weights_path, get_model_resunet_a, \
    resunet_a_weights_path

RUN_ON_CPU = True
WITH_TEST_TIME_AUGMENTATION = False
USE_ISTENMEZEJE = False
USE_RESUNET = False
PREDICTION_SIZE_MULTIPLIER = 1


def predict(image, model):
    predictions_smooth = predict_img_with_smooth_windowing(
        image,
        window_size=PATCH_SIZE,
        subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
        nb_classes=CLASS_COUNT,
        pred_func=(
            lambda img_batch_subdiv: model.predict((img_batch_subdiv))
        ),
        aug=WITH_TEST_TIME_AUGMENTATION
    )
    return from_one_hot_to_category_indices(predictions_smooth)


def plot():
    correct_pixel_percentage_unet, correct_pixel_percentage_resuneta, \
        m_iou_percent_unet, m_iou_percent_resuneta = \
        evaluate()

    if USE_RESUNET:
        fig, ax = plt.subplots(2, 2, figsize=[15, 12])
        test_img_ax = ax[0, 0]
        gt_ax = ax[0, 1]
        unet_ax = ax[1, 0]
    else:
        fig, ax = plt.subplots(1, 3, figsize=[16, 5])
        test_img_ax = ax[0]
        gt_ax = ax[1]
        unet_ax = ax[2]

    fig.suptitle("Prediction (patch size: {p}x{p})"
                 .format(p=PATCH_SIZE), fontsize=16)
    show(
        test_img.transpose([2, 0, 1])[:3, :, :],  # only the RGB
        ax=test_img_ax,
        title='original image'
    )
    show(ground_truth_image, ax=gt_ax, title='ground truth')

    unet_title = 'U-Net prediction, accuracy: {:.2f}%%'.format(correct_pixel_percentage_unet)
    show(
        predicted_image_unet,
        ax=unet_ax,
        title=unet_title
    )
    if USE_RESUNET:
        show(
            predicted_image_resuneta,
            ax=ax[1, 1],
            title='ResUNet-a prediction, accuracy: {a:.2f}%'
                .format(a=correct_pixel_percentage_resuneta)
        )
    plt.show()


def evaluate():
    correct_pixel_percentage_resuneta = 0
    m_iou_percent_resuneta = 0
    m_iou_percent_unet = 0

    ground_truth_categories_masked = ground_truth_categories[np.nonzero(ground_truth_categories)]

    if not RUN_ON_CPU:
        m_iou = MeanIoU(num_classes=CLASS_COUNT)
        m_iou.update_state(
            ground_truth_categories_masked,
            predicted_categories_unet[np.nonzero(ground_truth_categories)]
        )
        m_iou_percent_unet = m_iou.result().numpy() * 100.
        if USE_RESUNET:
            m_iou.update_state(
                ground_truth_categories_masked,
                predicted_categories_resuneta[np.nonzero(ground_truth_categories)]
            )
            m_iou_percent_resuneta = m_iou.result().numpy() * 100.

    total_pixel_count = ground_truth_categories_masked.size
    correct_pixel_count_unet = np.count_nonzero(
        predicted_categories_unet[np.nonzero(ground_truth_categories)] ==
        ground_truth_categories_masked
    )
    if USE_RESUNET:
        correct_pixel_count_resuneta = np.count_nonzero(
            predicted_categories_resuneta[np.nonzero(ground_truth_categories)] ==
            ground_truth_categories_masked
        )
        correct_pixel_percentage_resuneta = correct_pixel_count_resuneta / total_pixel_count * 100
    correct_pixel_percentage_unet = correct_pixel_count_unet / total_pixel_count * 100
    return correct_pixel_percentage_unet, correct_pixel_percentage_resuneta, \
        m_iou_percent_unet, m_iou_percent_resuneta


# Here we read in full images instead of just smaller patches of them.
# We might get out of memory error, so we need to read them one band at a time, that is why
#  we don't use the read logic we use for patches
def read_image(img_path):
    rasterio_reader = rasterio.open(img_path)
    img_tmp = []
    for i in range(1, 4):
        if USE_ISTENMEZEJE:
            img_tmp.append(read_one_band(rasterio_reader, i, prediction_size,
                                         scale_factor=DOWNSCALING_RATE ** 2))
        else:
            img_tmp.append(read_one_band(rasterio_reader, i, prediction_size))
    return normalize(img_tmp)


def read_mask(mask_path):
    ground_truth = rasterio.open(mask_path)
    if USE_ISTENMEZEJE:
        mask_one_dimension = \
            read_one_band(ground_truth, 1, prediction_size, scale_factor=DOWNSCALING_RATE ** 2)
    else:
        mask_one_dimension = np.array(cv2.imread(mask_path), dtype=np.uint8)[:, :, 0]

    if USE_ISTENMEZEJE:
        mask_one_dimension[mask_one_dimension == 5] = 0
        mask_one_dimension[mask_one_dimension == 4] = 5
        mask_one_dimension[mask_one_dimension == 2] = 4
        mask_one_dimension[mask_one_dimension == 3] = 2
        mask_one_dimension[mask_one_dimension == 5] = 3
    return mask_one_dimension


def add_alpha_and_elevation_channels():
    global test_img
    if USE_ELEVATION:
        test_img = add_channel(elevation, test_img)
    alpha = np.ones(ground_truth.shape)
    test_img = np.append(test_img, [alpha], axis=0)
    test_img = test_img.transpose([1, 2, 0])  # back to the shape we want


def run():
    global USE_ISTENMEZEJE, USE_RESUNET, prediction_size, ground_truth, test_img, elevation,\
        ground_truth_categories, ground_truth_image, predicted_categories_resuneta,\
        predicted_image_resuneta, predicted_categories_unet, predicted_image_unet
    prediction_size = PATCH_SIZE * PREDICTION_SIZE_MULTIPLIER

    if USE_ISTENMEZEJE:
        test = ['t_istenmezeje']
        dilation_rate = 10
    else:
        test = [  # ones with rails
            '471_5768',
            '472_5768',
            '473_5761',
            '473_5768',
            '473_5769',
            '473_5770',
            '474_5767',
            '474_5770',
            '474_5771',
            '474_5772',
            '475_5772',
            '475_5773',
            '476_5758',
            '476_5759',
            '476_5773',
            '476_5774',
            '477_5774',
            '477_5775',
            '477_5776',
            '477_5777',
            '477_5778',
            '466_5761',
            '466_5762',
            '467_5762',
            '467_5763',
            '468_5765',
        ]

    if USE_RESUNET:
        resuneta_model = get_model_resunet_a()
        resuneta_model.load_weights(resunet_a_weights_path)
    unet_model = get_model_unet()
    unet_model.load_weights(unet_weights_path)

    for test_id in test:
        print(test_id)
        img_path = '{}/{}_rgb.jp2'.format(IMGS_DIR, test_id)
        mask_path = '{}/{}_seg.tif'.format(MASKS_DIR, test_id)
        elevation_path = '{}/{}_dem.tif'.format(ELEVATION_DIR, test_id)

        ground_truth = change_some_categories(
            read_mask(mask_path)[:prediction_size, :prediction_size])
        test_img = read_image(img_path)
        elevation = read_elevation(elevation_path)[:prediction_size, :prediction_size]

        add_alpha_and_elevation_channels()

        (ground_truth_categories, ground_truth_image) = segmentation_map_from_mask(ground_truth)

        ground_truth_image = add_masked_alpha_channel(ground_truth_categories, ground_truth_image)

        if USE_RESUNET:
            predicted_resuneta = predict(test_img, resuneta_model)
        predicted_unet = unet_model.predict(np.expand_dims(test_img, 0))
        predicted_unet = np.argmax(np.squeeze(predicted_unet), -1)

        if USE_RESUNET:
            (predicted_categories_resuneta, predicted_image_resuneta) = \
                segmentation_map_from_mask(predicted_resuneta)
        (predicted_categories_unet, predicted_image_unet) = \
            segmentation_map_from_mask(predicted_unet)

        plot()


if __name__ == '__main__':
    # if we are training on GPU we might not have enough RAM on it to run this too,
    #   in this case we should use the CPU, otherwise using the GPU is faster
    if RUN_ON_CPU:
        config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
        sess = tf.compat.v1.Session(config=config)
        with tf.Graph().as_default():
            with sess:
                K.set_session(sess)
                run()
    else:  # run on GPU
        run()
