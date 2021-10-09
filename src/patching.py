import glob
import math
import os

import numpy as np
import tensorflow as tf
import tifffile as tiff
from patchify import patchify
from tensorflow.python.ops.image_ops_impl import ResizeMethod

from cfg import *
from data_handling import read_image, read_mask, read_elevation, convert_to_example


def save_example(original_image, ground_truth, elevation, writer):
    # patchify needs the image shape to be integer times the shape of the patch
    excess_height = original_image.shape[0] % PATCH_SIZE
    excess_width = original_image.shape[1] % PATCH_SIZE
    image = original_image[excess_height:, excess_width:]
    patches_img = patchify_image(image)
    patches_gt = patchify_image(ground_truth)
    patches_elevation = patchify_image(elevation)

    patch_count = 0
    for j in range(patches_img.shape[0]):
        for k in range(patches_img.shape[1]):
            gt_patch = patches_gt[j, k, 0, :, :, :]
            if np.max(gt_patch) > 0.:
                patch_count += 1
                original_image_patch = patches_img[j, k, 0, :, :, :]
                elevation_patch = patches_elevation[j, k, 0, :, :, :]
                example = convert_to_example(original_image_patch, elevation_patch, gt_patch)
                writer.write(example.SerializeToString())
    return patch_count


def patchify_image(image):
    if len(image.shape) == 2:
        image = np.expand_dims(image, -1)
    patches_img = patchify(image, (PATCH_SIZE, PATCH_SIZE, image.shape[2]), step=PATCH_SIZE)
    return patches_img


def write_file(image_name, j, k, parent_folder_name, patch):
    file_path = parent_folder_name + '/' + image_name + '_' + str(j) + '_' + str(k)
    file_path = file_path + ".tif"
    tiff.imwrite(file_path, patch, dtype=np.uint8)


# 'buildings' are interesting too, but there would not be enough of the other interesting
#    categories if we included it in the list below, and there is enough buildings
#    included this way anyway.
def is_gt_interesting(ground_truth):
    interesting_categories = set([CATEGORIES[x] for x in {'water', 'railway'}])
    has_interesting_category = \
        len(interesting_categories.intersection(np.unique(ground_truth))) > 0
    return has_interesting_category


# we're not interested in these distinctions atm, don't want to waste efforts learning them
def change_some_categories(ground_truth):
    ground_truth[ground_truth == ORIGINAL_CATEGORIES['ports']] = CATEGORIES['urban']

    # OLD_CATEGORIES['highway'] == CATEGORIES['roads'], we change highways to roads
    ground_truth[ground_truth == ORIGINAL_CATEGORIES['roads']] = ORIGINAL_CATEGORIES['highway']

    ground_truth[ground_truth == ORIGINAL_CATEGORIES['buildings']] = CATEGORIES['buildings']
    return ground_truth


def save_patches_for_usage_kind(
    img_paths,
    elevation_paths,
    gt_paths,
    usage_kind: str
):
    input_path = INPUT_PATH.format(usage_kind)
    if not os.path.exists(input_path):
        os.makedirs(input_path)

    print('Making and saving {} patches'.format(usage_kind))
    for i in range(math.ceil(len(img_paths) / float(IMAGE_COUNT_PER_RECORD))):
        img_path_subset = img_paths[i * IMAGE_COUNT_PER_RECORD: (i + 1) * IMAGE_COUNT_PER_RECORD]
        elevation_path_subset = \
            elevation_paths[i * IMAGE_COUNT_PER_RECORD: (i + 1) * IMAGE_COUNT_PER_RECORD]
        gt_path_subset = gt_paths[i * IMAGE_COUNT_PER_RECORD: (i + 1) * IMAGE_COUNT_PER_RECORD]
        save_patches_for_record(img_path_subset, elevation_path_subset, gt_path_subset, i,
                                input_path, usage_kind)


def save_patches_for_record(img_path_subset, elevation_path_subset, gt_path_subset, i, input_path,
                            usage_kind):
    from_file_name = get_file_name_without_extension_from_path(img_path_subset[0])[:-4]
    to_file_name = get_file_name_without_extension_from_path(img_path_subset[-1])[:-4]
    tmp_file_name = '{}/tmp.tfrec'.format(input_path)

    patch_count = 0
    with tf.io.TFRecordWriter(tmp_file_name) as writer:
        for j, (img_path, elevation_path, mask_path) in \
                enumerate(zip(img_path_subset, elevation_path_subset, gt_path_subset)):
            original_image_filename = get_file_name_without_extension_from_path(img_path)
            elevation_filename = get_file_name_without_extension_from_path(elevation_path)
            mask_filename = get_file_name_without_extension_from_path(mask_path)
            # sanity check: the appropriate original image, elevation and ground truth should
            #   be put together into an example
            assert original_image_filename[:-3] == mask_filename[:-3] and \
                   elevation_filename[:-3] == mask_filename[:-3]

            ground_truth = read_mask(mask_path)
            if not is_gt_interesting(ground_truth):
                continue
    
            ground_truth = change_some_categories(ground_truth)
            elevation = read_elevation(elevation_path)
            original_image = read_image(img_path)

            print('Processing {}, {} image {}'
                  .format(original_image_filename[:-4], usage_kind, i * IMAGE_COUNT_PER_RECORD + j))
            if PADDED_IMAGE_SIZE % PATCH_SIZE == 0:
                original_image = pad(original_image)
                ground_truth = pad(ground_truth)
                elevation = pad(elevation)
            patches_saved = save_example(original_image, ground_truth, elevation, writer)
            patch_count += patches_saved
    file_name = '{}/{}_{}_patches_{}-{}.tfrec'\
        .format(input_path, usage_kind, patch_count, from_file_name, to_file_name)
    os.rename(tmp_file_name, file_name)


def pad(patch):
    paddings = [[PAD_BEFORE_AFTER, PAD_BEFORE_AFTER], [PAD_BEFORE_AFTER, PAD_BEFORE_AFTER]]
    if len(patch.shape) > 2:
        paddings.append([0, 0])
    return tf.pad(patch, paddings).numpy()  # fill with 0 (masked out)
    # I did not want reflect padding, because, although it would increase the amount of content,
    #   it could produce false structures, like rails that have a sharp perpendicular turn


def get_file_name_without_extension_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]


def make_patches_if_necessary():
    if os.path.exists(PATCHES_PATH):
        return
    img_paths = glob.glob(os.path.join(IMGS_DIR, "*.jp2"))
    mask_paths = glob.glob(os.path.join(MASKS_DIR, "*.tif"))
    elevation_paths = glob.glob(os.path.join(ELEVATION_DIR, "*.tif"))

    img_paths.sort()
    elevation_paths.sort()
    mask_paths.sort()

    img_paths = img_paths[:USED_IMG_COUNT]
    elevation_paths = elevation_paths[:USED_IMG_COUNT]
    mask_paths = mask_paths[:USED_IMG_COUNT]

    # validation image count = test image count
    # we divide the rest which does not go into training into them equally
    train_image_count = int(len(img_paths) * TRAIN_RATIO)
    val_and_test_image_count = int((len(img_paths) - train_image_count) / 2)
    print('train_image_count: {}, val and test image_count each: {}'
          .format(train_image_count, val_and_test_image_count))

    save_patches_for_usage_kind(
        img_paths[train_image_count:train_image_count + val_and_test_image_count],
        elevation_paths[train_image_count:train_image_count + val_and_test_image_count],
        mask_paths[train_image_count:train_image_count + val_and_test_image_count],
        'val'
    )
    save_patches_for_usage_kind(
        img_paths[train_image_count + val_and_test_image_count:],
        elevation_paths[train_image_count + val_and_test_image_count:],
        mask_paths[train_image_count + val_and_test_image_count:],
        'test'
    )
    save_patches_for_usage_kind(
        img_paths[:train_image_count],
        elevation_paths[:train_image_count],
        mask_paths[:train_image_count],
        'train'
    )
