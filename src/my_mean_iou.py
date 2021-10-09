import tensorflow as tf


class MyMeanIOU(tf.keras.metrics.MeanIoU):

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(
            # we don't care about the masked_out "category" (index 0)
            tf.argmax(y_true[:, :, :, 1:], axis=-1),  # the 4th dimension is for the categories
            tf.argmax(y_pred[:, :, :, 1:], axis=-1),
            sample_weight)
