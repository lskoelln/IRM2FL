import numpy as np


'''TODO: write class'''

def PatchPercentile(pmin=90, pmax=99, clip=[0, 1], channel_last=True, inverted_image=False):

    def norm(x_):
        pmin_=np.percentile(x_, pmin)
        pmax_=np.percentile(x_, pmax)
        return (x_-pmin_)/(pmax_-pmin_)

    def _normalise(x, y=None):
        ### x must be array with 4 dim, y must be array with 3 dim
        ### x are patches, y is original image
        if inverted_image:
            x = np.abs(x-1)
            y = np.abs(y-1) if y is not None else y
        if channel_last:
            ns, _, _, nc = x.shape
        else:
            ns, nc, _, _ = x.shape
        for s in range(ns):
            for c in range(nc):
                if channel_last:
                    x[s,...,c] = norm(x[s,...,c])
                else:
                    x[s,c] = norm(x[s,c])
        if not clip is None:
            x = np.clip(x, *clip)
        if inverted_image:
            x = np.abs(x - 1)
        return x

    return _normalise

def Percentile(pmin=90, pmax=99, clip=[0, 1], channel_last=True, inverted_image=False):

    def norm(x_, pmin_=None, pmax_=None):
        if pmin_ is None: pmin_=np.percentile(x_, pmin)
        if pmax_ is None: pmax_=np.percentile(x_, pmax)
        return (x_-pmin_)/(pmax_-pmin_)

    def _normalise(x, y=None):
        ### x must be array with 4 dim, y must be array with 3 dim
        ### x are patches, y is original image
        if inverted_image:
            x = np.abs(x-1)
            y = np.abs(y-1) if y is not None else y
        if channel_last:
            ns, _, _, nc = x.shape
        else:
            ns, nc, _, _ = x.shape
        axis = (0, 1) if channel_last else (-2, -1)
        perc_min = np.percentile(y, pmin, axis=axis) if not y is None else [None] * nc
        perc_max = np.percentile(y, pmax, axis=axis) if not y is None else [None] * nc
        for s in range(ns):
            for c in range(nc):
                if channel_last:
                    x[s,...,c] = norm(x[s,...,c], perc_min[c], perc_max[c])
                else:
                    x[s,c] = norm(x[s,c], perc_min[c], perc_max[c])
        if not clip is None:
            x = np.clip(x, *clip)
        if inverted_image:
            x = np.abs(x-1)
        return x

    return _normalise

def Standardisation(mean=0, std=0.3, clip=[-1, 1], channel_last=True):

    def norm(x_, mean_=None, std_=None):
        if mean_ is None: mean_=np.mean(x_)
        if std_ is None: std_=np.std(x_)
        return ((x_ - mean_) / std_) * std + mean

    def _normalise(x, y=None):
        ### x must be array with 4 dim, y must be array with 3 dim
        ### x are patches, y is original image
        if channel_last:
            ns, _, _, nc = x.shape
        else:
            ns, nc, _, _ = x.shape
        axis = (0, 1) if channel_last else (-2, -1)
        mean_val = np.mean(y, axis=axis) if not y is None else [None] * nc
        std_val = np.std(y, axis=axis) if not y is None else [None] * nc
        for s in range(ns):
            for c in range(nc):
                if channel_last:
                    x[s,...,c] = norm(x[s,...,c], mean_val[c], std_val[c])
                else:
                    x[s,c] = norm(x[s,c], mean_val[c], std_val[c])
        if not clip is None:
            x = np.clip(x, *clip)
        return x
    return _normalise


def NoNorm():
    def _normalise(x, y=None):
        return x
    return _normalise