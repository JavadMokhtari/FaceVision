class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.stop_flag = False

    def __call__(self, train_loss, val_loss):
        if (val_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.stop_flag = True
        else:
            self.counter = 0
