import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from rasterio.plot import show
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.python.keras import backend as K

from cfg import BASIC_BATCH_SIZE, CLASS_COUNT, UNET_BATCH_MULTIPLIER, CONFUSION_MATRIX_DIR, \
    CATEGORIES, PATCH_SIZE, USE_UNET
from data_handling import from_one_hot_to_category_indices
from train import setup_input_stream, get_model_unet, unet_weights_path, get_model_resunet_a, \
    resunet_a_weights_path

# TODO: running on CPU stopped working, has a lot of problems from using the dataset to
# TODO |  needing tf ops instead of numpy
# this could help: https://www.tensorflow.org/api_docs/python/tf/compat/v1/data/Iterator#get_next
RUN_ON_CPU = False

EVALUATION_BATCH_SIZE = int(64 / BASIC_BATCH_SIZE)
if not USE_UNET:
    EVALUATION_BATCH_SIZE *= UNET_BATCH_MULTIPLIER


class Evaluator:
    def __init__(self, is_unet=True):
        if is_unet:
            self.model = get_model_unet()
            self.model.load_weights(unet_weights_path)
            self.batch_size = BASIC_BATCH_SIZE * UNET_BATCH_MULTIPLIER
        else:
            self.model = get_model_resunet_a()
            self.model.load_weights(resunet_a_weights_path)
            self.batch_size = BASIC_BATCH_SIZE
        self.test_sequence, self.test_iteration_count = \
            setup_input_stream('test', self.batch_size)
        self.test_iteration_count -= self.test_iteration_count % EVALUATION_BATCH_SIZE
        self.test_sequence = self.test_sequence.as_numpy_iterator()
        self.list_of_masked_ground_truth_categories = []
        self.masked_predictions = []
        self.confusion_matrices = []
        self.accuracies = []

    def run(self):
        # go through the test_sequence and compute the metrics
        # computing confusion matrix requires a lot of RAM, that is why we need to do a number of
        #    iterations of the main loop instead of just one loop. If we would have the
        #    EVALUATION_BATCH_SIZE a small number it could happen that there would be some
        #    category of which we would not see in any test image, and it would ruin our final
        #    confusion matrix, it would not have percentages as we'd like, because when a category
        #    is not seen in an iteration, it would have 0s in its row in the conf matrix.
        for k in range(self.test_iteration_count // EVALUATION_BATCH_SIZE):
            for i in range(EVALUATION_BATCH_SIZE):
                # both will have self.batch_size length of tensors
                image, ground_truth = next(self.test_sequence)
                # self.batch_size number of predictions
                prediction = self.model.predict(x=image, batch_size=self.batch_size)
                prediction = from_one_hot_to_category_indices(prediction)

                ground_truth_categories = from_one_hot_to_category_indices(ground_truth)
                ground_truth_categories_masked = \
                    ground_truth_categories[np.nonzero(ground_truth_categories)]
                masked_prediction = prediction[np.nonzero(ground_truth_categories)]
                self.list_of_masked_ground_truth_categories = \
                    np.append(self.list_of_masked_ground_truth_categories,
                              ground_truth_categories_masked)
                self.masked_predictions = np.append(self.masked_predictions, masked_prediction)
                self.compute_accuracy(ground_truth_categories_masked, i, masked_prediction, k)

                # self.plot(ground_truth_categories, prediction)

            self.compute_confusion_matrix()  # needs a lot of RAM!
            self.masked_predictions = []
            self.list_of_masked_ground_truth_categories = []
        self.display_results()

    def plot(self, ground_truth_categories, prediction):
        for i in range(ground_truth_categories.shape[0]):
            fig, ax = plt.subplots(1, 2, figsize=[12, 5])
            show(ground_truth_categories[i] / CLASS_COUNT, ax=ax[0], title='ground truth')

            show(
                prediction[i] / CLASS_COUNT,
                ax=ax[1],
                title='U-Net prediction'
            )
            plt.show()

    def display_results(self):
        average_acc = float(np.mean(self.accuracies, axis=0))
        print('average accuracy: {:.3f}'.format(average_acc))
        confusion_matrix = np.mean(self.confusion_matrices, axis=0)
        labels = list(CATEGORIES.keys())
        for i in range(CLASS_COUNT - 1):
            print('{}: {:.3f}'.format(labels[i], float(confusion_matrix[i, i])))
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(1, 1, 1)
        ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)\
            .plot(ax=ax, xticks_rotation=45., cmap='YlOrRd', values_format='.3f')
        file_name = '/cm_{date:%Y-%m-%d_%H:%M}_avg{avg}.png'\
            .format(date=datetime.now(), avg=round(average_acc))
        fig.savefig(CONFUSION_MATRIX_DIR + file_name)
        plt.show()

    def compute_confusion_matrix(self):
        confusion_matrix = metrics.confusion_matrix(
            self.list_of_masked_ground_truth_categories,
            self.masked_predictions,
            labels=range(1, CLASS_COUNT),
            normalize='true'
        ) * 100
        self.confusion_matrices.append(confusion_matrix)

    def compute_accuracy(self, masked_ground_truth_categories, i, masked_prediction,
                         test_iteration_index):
        total_pixel_count = masked_ground_truth_categories.size
        correct_pixel_count_unet = np.count_nonzero(
            masked_prediction == masked_ground_truth_categories
        )
        accuracy = correct_pixel_count_unet / total_pixel_count * 100
        self.accuracies.append(accuracy)
        print('{}      ({}/{})'.format(
            accuracy,
            i + test_iteration_index * EVALUATION_BATCH_SIZE + 1,
            self.test_iteration_count
        ))


if __name__ == '__main__':
    if not os.path.exists(CONFUSION_MATRIX_DIR):
        os.makedirs(CONFUSION_MATRIX_DIR)
    # if we are training on GPU we might not have enough RAM on it to run this too,
    #   in this case we should use the CPU, otherwise using the GPU is faster
    if RUN_ON_CPU:
        config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
        sess = tf.compat.v1.Session(config=config)
        with tf.Graph().as_default():
            with sess:
                K.set_session(sess)
                Evaluator(is_unet=USE_UNET).run()
    else:  # run on GPU
        Evaluator(is_unet=USE_UNET).run()
