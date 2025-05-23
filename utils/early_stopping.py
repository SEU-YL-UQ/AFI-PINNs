import numpy as np


class EarlyStopping:
    """ https://github.com/Bjarten/early-stopping-pytorch """
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, delta=1E-7):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def reset(self):
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        

    def __call__(self, val_loss):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score - self.delta:
            self.counter += 1
            # self.best_score = score
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # reset counter if loss is reducing
            self.best_score = score
            self.counter = 0

