import skorch

#
# Note: the 'break' is generated by an 'KeyboardInterrupt'
#

class EarlyStopping(skorch.callbacks.EarlyStopping):
    """
    Extends skorch.callbacks.EarlyStopping adding 'warmup': number of epochs
    to wait until early stopping.
    """

    def __init__(self, warmup=0, patience=5, threshold=1e-4, **kwargs):
        super().__init__(patience=patience, threshold=threshold, **kwargs)
        self.warmup = warmup
        self._epoch = 0

    def on_epoch_end(self, net, **kwargs):
        self._epoch += 1
        if self._epoch >= self.warmup:
            super().on_epoch_end(net, **kwargs)
    # end
# end