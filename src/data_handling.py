import math
import sys

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image
from numpy import ndarray
from rasterio import DatasetReader
from rasterio.enums import Resampling

from cfg import CLASS_COUNT, PATCH_SIZE, USE_ELEVATION, \
    N_BANDS, CATEGORY_COLORS, GROUND_TRUTH_SHAPE

DOWNSCALING_RATE = 1.

ORIGINAL_IMAGE_PATCH_KEY = 'original'
ELEVATION_KEY = 'normalized_elevation'
GT_KEY = 'gt'
TFREC_FORMAT = {
    ORIGINAL_IMAGE_PATCH_KEY: tf.io.FixedLenFeature([], tf.string),  # tf.string means bytestring
    ELEVATION_KEY: tf.io.FixedLenFeature([], tf.string),  # tf.string means bytestring
    GT_KEY: tf.io.FixedLenFeature([], tf.string),   # shape [] means single element
}


# we can read bigger images one band at a time
def read_one_band(
    rasterio_reader: DatasetReader,
    band_id,
    crop_to=sys.maxsize,
    scale_factor=1. / DOWNSCALING_RATE
):
    # resample data to target shape
    data = rasterio_reader.read(
        band_id,
        out_shape=(
            1,
            int(rasterio_reader.height * scale_factor),
            int(rasterio_reader.width * scale_factor)
        ),
        resampling=Resampling.mode
    )
    return np.array(data[:crop_to, :crop_to])


def from_one_hot_to_category_indices(data: ndarray):
    """ converts data (numpy array) that has one-hot encoded categories in its last dimension
        to category indices (with the other dimensions remaining) """
    return np.argmax(data, axis=-1)


def read_image(img_path):
    return np.array(cv2.imread(img_path), dtype=np.uint8)


def read_mask(mask_path):
    return np.array(cv2.imread(mask_path), dtype=np.uint8)[:, :, 0]


def normalize(img):
    return np.array(img, dtype=np.float32) / 255.


# with per-image normalization
def read_elevation(elevation_path):
    e = np.array(Image.open(elevation_path))
    min_elevation = np.min(e)
    e = (e - min_elevation) / (np.max(e) - min_elevation)
    return e


def segmentation_map_from_mask(categories):
    image = np.zeros((categories.shape[0], categories.shape[1], 3), dtype=np.float32)
    for i in range(categories.shape[0]):
        for j in range(categories.shape[1]):
            for k in range(3):
                image[i, j, k] = CATEGORY_COLORS[categories[i, j]][k]
    image = image.transpose([2, 0, 1])
    return categories, normalize(image)


def add_channel(channel, original_image):
    return np.append(original_image, [channel], axis=0)


def add_masked_alpha_channel(mask, original_image):
    alpha = np.zeros(mask.shape, dtype=np.float32)
    alpha[np.nonzero(mask)] = 1.
    return add_channel(alpha, original_image)


def add_channel_tf(channel, original_image):
    return tf.concat([original_image, channel], axis=-1)


def add_masked_alpha_channel_tf(ground_truth, original_image):
    alpha = tf.where(tf.not_equal(ground_truth, tf.constant(0.)), ones, zeros)
    return add_channel_tf(alpha, original_image)


def convert_to_example(original_patch, elevation, gt):
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                ORIGINAL_IMAGE_PATCH_KEY: bytes_feature(tf.io.serialize_tensor(original_patch)),
                ELEVATION_KEY: bytes_feature(tf.io.serialize_tensor(elevation)),
                GT_KEY: bytes_feature(tf.io.serialize_tensor(gt))
            }
        )
    )


def parse_example(example_proto):
    # Parse the input tf.Example proto using the dictionary TFREC_FORMAT
    features = tf.io.parse_single_example(example_proto, TFREC_FORMAT)

    ground_truth = features[GT_KEY]
    ground_truth = tf.io.parse_tensor(ground_truth, tf.uint8)
    ground_truth = tf.reshape(ground_truth, GROUND_TRUTH_SHAPE)
    mask = tf.cast(ground_truth, dtype=tf.float32)

    original_image_patch = features[ORIGINAL_IMAGE_PATCH_KEY]
    original_image_patch = tf.cast(
        tf.io.parse_tensor(original_image_patch, tf.uint8),
        dtype=tf.float32)

    elevation_patch = features[ELEVATION_KEY]
    elevation_patch = tf.io.parse_tensor(elevation_patch, tf.float32)
    elevation_patch = tf.reshape(elevation_patch, GROUND_TRUTH_SHAPE)

    if USE_ELEVATION:
        original_image_patch = add_channel_tf(elevation_patch, original_image_patch)
    original_image_patch = add_masked_alpha_channel_tf(mask, original_image_patch)
    original_image_patch = tf.reshape(original_image_patch, [PATCH_SIZE, PATCH_SIZE, N_BANDS])

    ground_truth = tf.one_hot(ground_truth, CLASS_COUNT, dtype=tf.float32)
    ground_truth = tf.reshape(ground_truth, [PATCH_SIZE, PATCH_SIZE, CLASS_COUNT])

    return original_image_patch, ground_truth


def parse_example_factory_with_augmentation(crop_boxes):
    global zeros, ones
    zeros = tf.zeros(GROUND_TRUTH_SHAPE, dtype=tf.float32)
    ones = tf.ones(GROUND_TRUTH_SHAPE, dtype=tf.float32)

    def inner_parse_example(example_proto):
        patch_img, patch_mask = parse_example(example_proto)
        return augment(patch_img, patch_mask, crop_boxes)

    return inner_parse_example


def parse_example_factory():
    global zeros, ones
    zeros = tf.zeros(GROUND_TRUTH_SHAPE, dtype=tf.float32)
    ones = tf.ones(GROUND_TRUTH_SHAPE, dtype=tf.float32)

    def inner_parse_example(example_proto):
        return parse_example(example_proto)

    return inner_parse_example


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.numpy()]))


def transform_both(op, patch_img, patch_mask, args={}):
    return op(patch_img, **args), op(patch_mask, **args)


def augment(patch_img, patch_mask, crop_boxes):
    patch_img, patch_mask = random_rotate(patch_img, patch_mask)
    patch_img, patch_mask = random_zoom_in(patch_img, patch_mask, crop_boxes)
    return tf.squeeze(patch_img), tf.squeeze(patch_mask)


def random_rotate(patch_img, patch_mask):
    random_angle = tf.random.uniform([], 0, 1, dtype=tf.float32)
    patch_img, patch_mask = transform_both(tfa.image.rotate,  # fill mode: put zeros (masked out)
       patch_img, patch_mask,
       {'angles': random_angle * math.pi, 'interpolation': 'nearest'})
    return patch_img, patch_mask


def random_zoom_in(patch_img, patch_mask, crop_boxes):
    random_zoom_ind = tf.random.uniform([], 0, len(crop_boxes), dtype=tf.int32)
    random_crop_box = [tf.constant(np.array(crop_boxes), dtype=tf.float32)[random_zoom_ind]]

    patch_img = tf.image.crop_and_resize(
        tf.expand_dims(patch_img, 0),  # needs 4D tensor
        random_crop_box,
        box_indices=tf.zeros(1, dtype=tf.int32),
        crop_size=(PATCH_SIZE, PATCH_SIZE))
    patch_mask = tf.image.crop_and_resize(
        tf.expand_dims(patch_mask, 0),
        random_crop_box,
        box_indices=tf.zeros(1, dtype=tf.int32),
        crop_size=(PATCH_SIZE, PATCH_SIZE),
        method='nearest')  # we don't want to introduce invalid labels with interpolated values
    return patch_img, patch_mask
