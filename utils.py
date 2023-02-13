import numpy as np


class EarlyStopper:
    """
    early stopping the training based on
    validation loss
    Parameters
    patience: Number of epochs with no improvement after which training will be stopped.
    min_delta: an absolute change of less than min_delta, will count as no improvement.

    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
