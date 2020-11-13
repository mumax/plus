class FieldQuantity:

    def __init__(self, impl):
        self._impl = impl

    @property
    def name(self):
        return self._impl.name

    @property
    def unit(self):
        return self._impl.unit

    @property
    def ncomp(self):
        return self._impl.ncomp

    def eval(self):
        return self._impl.eval()

    def __call__(self):
        return self.eval()

    def average(self):
        return self._impl.average()

    def _bench(self, ntimes=100):
        import time
        start = time.time()
        for i in range(ntimes):
            self._impl.exec()
        stop = time.time()
        return (stop-start)/ntimes
