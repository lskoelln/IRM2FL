from scipy import interpolate
import numpy as np

class nested_preprocessing:
    def __init__(self, list_functions=None):
        self.list_functions = list_functions
    def __call__(self, x):
        self.call(x)
    def call(self, x):
        for _func in self.list_functions:
            x = _func(x)
        return x

def Interpolation(new_shape=None, channel_first=True, dtype=np.float32):

    def interpolate2d(array):
        (xini, yini) = array.shape
        x = np.linspace(1, xini, xini, dtype=int)
        y = np.linspace(1, yini, yini, dtype=int)
        xnew = np.linspace(1, xini, new_shape[0], dtype=int)
        ynew = np.linspace(1, yini, new_shape[1], dtype=int)
        f = interpolate.RectBivariateSpline(x, y, array, bbox=[None, None, None, None], kx=3, ky=3, s=0)
        return f(xnew, ynew)

    def _preprocess(x, **kwargs):
        ### requires 3-channel input x
        if channel_first:
            nch, old_shape = x.shape[0], x.shape[1:]
        else:
            old_shape, nch = x.shape[:-1], x.shape[-1]
        if old_shape==new_shape:
            return x
        output = np.zeros((nch,)+new_shape) if channel_first else np.zeros(new_shape+(nch,))
        for n in range(nch):
            if channel_first:
                output[n] = interpolate2d(x[n])
            else:
                output[...,n] = interpolate2d(x[...,n])
        return dtype(output)

    return _preprocess