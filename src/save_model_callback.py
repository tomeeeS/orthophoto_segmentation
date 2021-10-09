import pickle

import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import Callback

# this save is slower than the hdf5 method, we don't use this every time, only every x-th time
SAVE_EVERY_XTH = 5


class SaveModelCallback(Callback):

    def __init__(self, model: Model, tf_checkpoint_path: str, monitored_metric='val_loss',
                 mode='min', pickle_path: str = None, starting_val_loss=None):
        super().__init__()
        self.model = model
        self.tf_checkpoint_path = tf_checkpoint_path
        if pickle_path is not None:
            self.pickle_path = pickle_path
        else:
            self.pickle_path = tf_checkpoint_path + '_pickle'
        self.best_val_loss = starting_val_loss
        self.save_locally = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
        self.monitored_metric = monitored_metric
        self.mode = mode
        self.better_one_seen_counter = 0

    def on_epoch_end(self, epoch, logs=None):
        # self.model.save_weights(self.tf_checkpoint_path + '_last', options=self.save_locally, )
        if self.best_val_loss is None:
            is_better = True
        else:
            if self.mode == 'max':
                is_better = logs[self.monitored_metric] > self.best_val_loss
            else:
                is_better = logs[self.monitored_metric] < self.best_val_loss
        if is_better:
            self.best_val_loss = logs[self.monitored_metric]
            self.better_one_seen_counter += 1
            if self.better_one_seen_counter == SAVE_EVERY_XTH:
                self.better_one_seen_counter = 0
                print('saving model')
                self.model.save_weights(self.tf_checkpoint_path, options=self.save_locally)
                self.model.save(self.tf_checkpoint_path + '_whole', options=self.save_locally)
                # pickle_file = open(self.pickle_path, 'wb')
                # pickle.dump(self.model, pickle_file)
