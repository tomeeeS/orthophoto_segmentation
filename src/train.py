# data to be downloaded from: https://ieee-dataport.org/open-access/geonrw

import glob
import os.path
from functools import reduce

import numpy as np
import segmentation_models as sm
import tensorflow as tf
from keras_unet_collection.model_resunet_a_2d import resunet_a_2d
from matplotlib import pyplot as plt
from rasterio.plot import show
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, \
    CSVLogger, ModelCheckpoint
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.optimizer_v2.adam import Adam

from cfg import *
from losses import bce_dice_loss, weighted_binary_crossentropy, \
    focal_wbce_dice_loss
from data_handling import parse_example_factory_with_augmentation, segmentation_map_from_mask,\
    parse_example_factory
from patching import make_patches_if_necessary
from save_model_callback import SaveModelCallback
from unet_model import unet_model

# use previous weights and continue from there, or restart and delete previous state
CONTINUE_LEARNING = False

N_EPOCHS = 10000
EARLY_STOPPING_PATIENCE = 40

patch_size_text = '{}'.format(PATCH_SIZE)
resunet_a_weights_path = 'weights_resunet_a_p' + patch_size_text
unet_weights_path = 'weights_unet_p' + patch_size_text
if not os.path.exists(resunet_a_weights_path):
    os.makedirs(resunet_a_weights_path)
if not os.path.exists(unet_weights_path):
    os.makedirs(unet_weights_path)
resunet_a_weights_path += '/weights.hdf5'
unet_weights_path += '/weights.hdf5'


def compile_model(model):
    mean_iou = sm.metrics.IOUScore(class_indexes=range(1, CLASS_COUNT))
    rails_iou = sm.metrics.IOUScore(class_indexes=[6], name='rails_iou')
    # about class_indexes: we don't care about the masked_out "category" (index 0)

    # from tensorflow.keras.optimizers import Adam    or
    # from tensorflow.python.keras.optimizer_v2.adam import Adam
    # Adam(learning_rate=x)   or 'adam' if that doesn't work
    model.compile(optimizer='adam', loss=focal_wbce_dice_loss,
                  metrics=[mean_iou, rails_iou])
    return model


def get_model_unet():
    model = unet_model(CLASS_COUNT, PATCH_SIZE, n_channels=N_BANDS, upconv=True)
    return compile_model(model)


def get_model_resunet_a():
    batch_norm = BASIC_BATCH_SIZE >= 2
    model = resunet_a_2d((PATCH_SIZE, PATCH_SIZE, N_BANDS), [32, 64, 128, 256, 512, 1024],
                         dilation_num=[1, 3, 15, 31],
                         n_labels=CLASS_COUNT,
                         activation='ReLU', output_activation='Sigmoid',
                         batch_norm=batch_norm, unpool=True, name='resunet')
    return compile_model(model)


def count_available_inputs(tfrecord_paths: [str]):
    add_inputs_count_from_path = lambda i, path: i + int(path.split('_')[1])
    return reduce(add_inputs_count_from_path, tfrecord_paths, 0)


def setup_input_stream(
    usage_kind: str,
    batch_size=BASIC_BATCH_SIZE,
):
    input_path = INPUT_PATH.format(usage_kind)
    tfrecord_paths = glob.glob(os.path.join(input_path, "*.tfrec"))
    inputs_used = int(count_available_inputs(tfrecord_paths) / DATA_DIVIDER)
    print('{} inputs used: {}'.format(usage_kind, inputs_used))

    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    if usage_kind == 'train':
        crop_boxes = make_crop_boxes_for_random_zoom_in()
        parser = parse_example_factory_with_augmentation(crop_boxes)
    else:
        parser = parse_example_factory()
    dataset = tf.data.TFRecordDataset(tfrecord_paths, num_parallel_reads=tf.data.AUTOTUNE)\
        .with_options(ignore_order)\
        .take(inputs_used)\
        .map(parser, num_parallel_calls=tf.data.AUTOTUNE)\
        .repeat()\
        .batch(batch_size, drop_remainder=True)\
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    if usage_kind == 'train':
        dataset = dataset.shuffle(buffer_size=50, reshuffle_each_iteration=True)
    return dataset, inputs_used


def make_crop_boxes_for_random_zoom_in():
    scales = list(np.arange(0.52, 1.0, 0.04))
    variation_count = 3
    crop_boxes = []  # will contain not centered boxes too
    for scale in scales:
        for j in range(variation_count):
            for k in range(variation_count):
                zoom_in_amount = 1. - scale
                x1 = 0. + j * (zoom_in_amount / 2.)
                y1 = 0. + k * (zoom_in_amount / 2.)
                x2 = x1 + scale
                y2 = y1 + scale
                crop_boxes.append([x1, y1, x2, y2])
    return crop_boxes


def train_net(is_unet):
    if is_unet:
        weights_path = unet_weights_path
        model = get_model_unet()
        model_title = 'U-Net'
        batch_size = UNET_BATCH_SIZE
    else:  # ResUNet-a
        weights_path = resunet_a_weights_path
        model = get_model_resunet_a()
        model_title = 'ResUNet-a'
        batch_size = BASIC_BATCH_SIZE

    train_inputs, train_inputs_used = setup_input_stream('train', batch_size)
    # plot_augmented_patches(train_inputs)
    val_inputs, val_inputs_used = setup_input_stream('val', batch_size)

    train_steps_per_epoch = int(train_inputs_used / batch_size)
    print("Starting training {}, batch size: {}, patch size: {}x{}, training steps count: {}"
          .format(model_title, batch_size, PATCH_SIZE, PATCH_SIZE, train_steps_per_epoch))

    if CONTINUE_LEARNING and os.path.isfile(weights_path):
        print('LOADING WEIGHTS')
        model.load_weights(weights_path)

    callbacks = assemble_callbacks(weights_path, model)

    model.fit(train_inputs,
              batch_size=batch_size,
              epochs=N_EPOCHS,
              verbose=1,
              shuffle=True,
              callbacks=callbacks,
              validation_data=val_inputs,
              steps_per_epoch=train_steps_per_epoch,
              validation_steps=int(val_inputs_used / batch_size),
              )
    return model


def assemble_callbacks(weights_path, model):
    early_stopping = EarlyStopping(monitor=MONITORED_METRIC, mode='max', min_delta=0,
                                   patience=EARLY_STOPPING_PATIENCE, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.33, patience=5)
    model_checkpoint = SaveModelCallback(model, weights_path, MONITORED_METRIC, mode='max')
    model_checkpoint2 = ModelCheckpoint(weights_path, save_best_only=True,
                                       monitor=MONITORED_METRIC, mode='max')
    csv_logger = CSVLogger('output/log_unet.csv', append=True, separator=';')
    callbacks = [model_checkpoint, model_checkpoint2, csv_logger, reduce_lr, early_stopping]
    return callbacks


def plot_augmented_patches(train_inputs):
    test_img, gt = next(iter(train_inputs))
    test_img = np.array(test_img)
    gt = np.array(gt)
    for i in range(test_img.shape[0]):
        fig, ax = plt.subplots(1, 2, figsize=[12, 5])
        gt_labels = np.argmax(gt[i], -1)
        ground_truth_categories, ground_truth_image = segmentation_map_from_mask(gt_labels)
        show(
            test_img[i].transpose([2, 0, 1])[:3, :, :],  # only the RGB
            ax=ax[0],
            title='original image'
        )
        show(ground_truth_image, ax=ax[1], title='ground truth')
        plt.show()


if __name__ == '__main__':
    make_patches_if_necessary()

    train_net(is_unet=USE_UNET)
