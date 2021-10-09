from typing import Callable, Union
import numpy as np
import tensorflow as tf
import segmentation_models as sm

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.losses import binary_crossentropy
from functools import partial, update_wrapper

from cfg import CLASS_COUNT, BASIC_BATCH_SIZE, PATCH_SIZE, CLASS_WEIGHTS


# taken from: https://github.com/maxvfischer/keras-image-segmentation-loss-functions/blob/master/
#   losses/multiclass_losses.py
# but according to the paper referenced below, weights should be calculated from the current
#   true positives
def tanimoto_loss(class_weights=1.) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    Tanimoto loss. Defined in the paper "ResUNet-a: a deep learning framework for
    semantic segmentation of remotely sensed data", under 3.2.4. Generalization to multiclass
    imbalanced problems. See https://arxiv.org/pdf/1904.00592.pdf Used as loss function for
    multi-class image segmentation with one-hot encoded masks.

    :return: Tanimoto  loss function (Callable[[tf.Tensor, tf.Tensor], tf.Tensor])
    """
    if not isinstance(class_weights, tf.Tensor):
        class_weights = tf.constant(class_weights)

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute Tanimoto loss.
        :param y_true: True masks (tf.Tensor,
            shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
        :param y_pred:
            Predicted masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>,
        <N_CLASSES>))
        :return: Tanimoto loss (tf.Tensor, shape=(None, ))
        """
        axis_to_reduce = range(1, K.ndim(y_pred))  # All axis but first (batch)
        numerator = y_true * y_pred * class_weights
        numerator = K.sum(numerator, axis=axis_to_reduce)

        denominator = (y_true**2 + y_pred**2 - y_true * y_pred) * class_weights
        denominator = K.sum(denominator, axis=axis_to_reduce)
        return 1 - numerator / denominator

    return loss


def tanimoto_with_complements(class_weights=1.) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    From "ResUNet-a: a deep learning framework for semantic segmentation of remotely sensed data"
    """
    tanimoto = tanimoto_loss(class_weights)

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        normal = tanimoto(y_true, y_pred)
        complement = tanimoto(1. - y_true, 1. - y_pred)
        return (normal + complement) / 2

    return loss


# taken from:
#   https://github.com/umbertogriffo/focal-loss-keras/blob/master/src/loss_function/losses.py
# never managed to make it work
def categorical_focal_loss(alpha, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed


# closure for initializing variables the returned loss function can use with keeping its signature
# taken from: https://github.com/maxvfischer/keras-image-segmentation-loss-functions/blob/master/
#   losses/multiclass_losses.py
def multiclass_focal_loss(class_weights: Union[list, np.ndarray, tf.Tensor],
                          gamma: Union[list, np.ndarray, tf.Tensor]
                          ) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    Focal loss.
        FL(p, p̂) = -∑class_weights*(1-p̂)ᵞ*p*log(p̂)
    Used as loss function for multi-class image segmentation with one-hot encoded masks.
    :param class_weights: Class weight coefficients (Union[list, np.ndarray, tf.Tensor], len=<N_CLASSES>)
    :param gamma: Focusing parameters, γ_i ≥ 0 (Union[list, np.ndarray, tf.Tensor], len=<N_CLASSES>)
    :return: Focal loss function (Callable[[tf.Tensor, tf.Tensor], tf.Tensor])
    """
    if not isinstance(class_weights, tf.Tensor):
        class_weights = tf.constant(class_weights)
    if not isinstance(gamma, tf.Tensor):
        gamma = tf.constant(gamma)

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute focal loss.
        :param y_true: True masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
        :param y_pred: Predicted masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
        :return: Focal loss (tf.Tensor, shape=(None,))
        """
        f_loss = -(class_weights * (1-y_pred)**gamma * y_true * K.log(y_pred))

        # Average over each data point/image in batch
        axis_to_reduce = range(1, K.ndim(f_loss))
        f_loss = K.mean(f_loss, axis=axis_to_reduce)

        return f_loss

    return loss


def bce_dice_loss(y_true, y_pred):
    return (binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)) / 2.


def focal_wbce_dice_loss(y_true, y_pred):
    return weighted_binary_crossentropy()(y_true, y_pred) * 0.5 + \
        sm.losses.binary_focal_loss(y_true, y_pred) * 0.3 + \
        dice_loss(y_true, y_pred) * 0.2


def dice_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# taken from: https://github.com/maxvfischer/keras-image-segmentation-loss-functions/blob/master/
#   losses/multiclass_losses.py
def multiclass_weighted_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Compute weighted Dice loss.
    :param y_true: True masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
    :param y_pred: Predicted masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
    :return: Weighted Dice loss (tf.Tensor, shape=(None,))
    """
    axis_to_reduce = range(1, K.ndim(y_pred))  # Reduce all axis but first (batch)
    numerator = y_true * y_pred * CLASS_WEIGHTS  # Broadcasting
    numerator = 2. * K.sum(numerator, axis=axis_to_reduce)

    denominator = (y_true + y_pred) * CLASS_WEIGHTS  # Broadcasting
    denominator = K.sum(denominator, axis=axis_to_reduce)

    return 1 - numerator / denominator


# taken from: https://github.com/akensert/keras-weighted-multilabel-binary-crossentropy/blob
#   /master/wbce.py
def weighted_binary_crossentropy():
    """A weighted binary crossentropy loss function
    that works for multilabel classification
    """

    # The below is needed to be able to work with keras' model.compile()
    def wrapped_partial(func, *args, **kwargs):
        partial_func = partial(func, *args, **kwargs)
        update_wrapper(partial_func, func)
        return partial_func

    def wrapped_weighted_binary_crossentropy(y_true, y_pred, class_weights):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0-K.epsilon())
        # cross-entropy loss with weighting
        out = -(y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred))

        return K.mean(out * class_weights, axis=-1)

    return wrapped_partial(wrapped_weighted_binary_crossentropy, class_weights=CLASS_WEIGHTS)


if __name__ == '__main__':
    y_t = K.constant([[1., 0.], [0., 1.]])
    y_p = K.constant([[0.51, 0.49], [0.6, 0.4]])
    print(dice_coef(y_t, y_p).numpy())
