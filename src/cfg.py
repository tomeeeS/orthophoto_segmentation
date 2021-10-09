# constants pertaining to the size of the utilized data (PATCH_SIZE, BATCH_SIZE)
#  are optimized for 6 GB RAM

import sys

USE_UNET = False

# - if we want to hasten the training (or evaluating) process, we can specify a number
#    to divide the amount of available inputs with.
DATA_DIVIDER = 50

PATCH_SIZE = 256  # should be divisible by 32 (based on the models used)
ORIGINAL_IMAGE_SIZE = 1000
PADDED_IMAGE_SIZE = 1024
PAD_BEFORE_AFTER = (PADDED_IMAGE_SIZE - ORIGINAL_IMAGE_SIZE) // 2
GROUND_TRUTH_SHAPE = [PATCH_SIZE, PATCH_SIZE, 1]

BASIC_BATCH_SIZE = 1
# U-Net uses less RAM than ResUNet-a, can handle more patches at once
UNET_BATCH_MULTIPLIER = 4
UNET_BATCH_SIZE = BASIC_BATCH_SIZE * UNET_BATCH_MULTIPLIER

USE_ELEVATION = True

if USE_ELEVATION:
    N_BANDS = 5  # RGB, elevation, alpha (because we use masking)
else:
    N_BANDS = 4
CLASS_COUNT = 9

USED_IMG_COUNT = 7356  # out of 7356 images. lower it to save disk space with patch generation

TRAIN_RATIO = 0.85  # ratio of inputs used for training, out of all the inputs (=1)
VALIDATION_RATIO = 0.08

MONITORED_METRIC = 'val_iou_score'

IMAGE_COUNT_PER_RECORD = USED_IMG_COUNT // (6 * 16)
# this way the number of training tfrecords used will be divisible by 16 (good for TPU training on
#   Kaggle according to docs). And there will be 8 validation tfrec files, one for each TPU core

if PATCH_SIZE <= 256:
    PATCH_PER_IMAGE = 16
else:
    PATCH_PER_IMAGE = 4

PATH_TO_DATA = 'data'
IMGS_DIR = PATH_TO_DATA + "/images"
ELEVATION_DIR = PATH_TO_DATA + "/elevation"
MASKS_DIR = PATH_TO_DATA + "/masks"
PATCHES_PATH = PATH_TO_DATA + '/patches/p{}'.format(PATCH_SIZE)
INPUT_PATH = PATCHES_PATH + '/{}'
OUTPUT_DIR = "output"
CONFUSION_MATRIX_DIR = OUTPUT_DIR + '/confusion_matrix'

CATEGORY_COLORS = {
    0: [255, 255, 255],  # unclassified
    1: [27, 120, 55],  # woodland
    2: [21, 47, 153],  # water
    3: [140, 86, 75],  # brown for agricultural
    4: [127, 127, 127],  # gray urban
    5: [123, 224, 123],  # olive for grassland
    6: [255, 127, 14],  # orange for railway
    7: [223, 194, 125],  # roads
    8: [0, 0, 0],  # buildings
}

CATEGORIES = {
    'woodland': 1,
    'water': 2,
    'agricultural': 3,
    'urban': 4,
    'grassland': 5,
    'railway': 6,
    'roads': 7,
    'buildings': 8,
}

ORIGINAL_CATEGORIES = {
    'woodland': 1,
    'water': 2,
    'agricultural': 3,
    'urban': 4,
    'grassland': 5,
    'railway': 6,
    'highway': 7,
    'ports': 8,
    'roads': 9,
    'buildings': 10,
}

# calculated percentages with geonrw/calc_and_plot_stats.py for my manipulated dataset
#   and then inverted the numbers and divided them with their sum to normalize to 1 again
#   and finally gave a positive weight to masked_out too
CLASS_WEIGHTS = [
    0.1,  # masked_out
    0.119647703286762,  # 'woodland'
    0.010842798271183,  # 'water'
    0.063775253794351,  # 'agricultural'
    0.049502462559051,  # 'urban'
    0.142727912353,     # 'grassland'
    0.27301738868228,   # 'railway'
    0.141094582370087,  # 'roads'
    0.199391898683285,  # 'buildings'
]


def is_debug():
    return sys.gettrace() is not None
