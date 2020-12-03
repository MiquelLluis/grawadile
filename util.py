#!/usr/bin/env python3
#
# util.py
#
# Utility functions.
#

import itertools
import os
import time

import numpy as np
from matplotlib import pyplot as plt


try:
    del print
except:
    pass
_print = print
def print(*args, dest='both', sep=' ', end='\n', **kw_args):
    """Print messages to the console and/or to the notebook."""
    dest = dest.lower()
    if dest in ('print', 'both'):
        _print(*args, sep=sep, end=end, **kw_args)
    if dest in ('terminal', 'both'):
        bstring = sep.join([repr(s) for s in args]) + end
        if not isinstance(bstring, bytes):
            bstring = bstring.encode()
        os.write(1, bstring)


def register(msg):
    """Register a message into a predefined register file."""
    _file = 'register.txt'
    with open(_file, 'a') as f:
        f.write(time.strftime("%Y%m%d,%H%M%S ") + msg + '\n')


def clearprint():
    print(" "*80, end='\r')


def closest_pow2(x):
    """Returns the closest power of 2 to 'x'."""
    if not isinstance(x, int) or (x <= 0):
        raise ValueError("'x' must be a positive natural number")
    
    return 2 ** int(round(np.log2(x)))


def next_pow2(x):
    """Returns the next closest power of 2 to 'x'."""
    if not isinstance(x, int) or (x <= 0):
        raise ValueError("'x' must be a positive natural number")
    
    return 2 ** int(np.ceil(np.log2(x)))


def _mif(x):
    """Minimum integer factor."""
    if isinstance(x, float) and not x.is_integer():
        raise ValueError("'x' must be an integer >= 2")
    if x % 2 == 0:
        return 2
    for n in range(3, int(x/2)+1, 2):
        if x % n == 0:
            return n
_mif = np.vectorize(_mif, otypes=[int])
def mif(x):
    """Minimum integer factor."""
    res = np.atleast_1d(_mif(x))
    if len(res) == 1:
        res = res[0]
    return res


def pad_centered(x, length):
    """Pads an array with zeros.

    Similar to numpy.pad(x, pad_width, 'constant'), but specifying the total
    length instead. Allows asymmetric pad when len(x) is odd.
    
    PARAMETERS
    ----------
    x: array_like (1d)
        Array to pad.

    length: int
        Length of the target array.

    RETURNS
    -------
    y: array_1d
        Array zero-padded.

    off: int
        Offset (position of x[0] in y).

    """
    x = np.asarray(x)
    lx = len(x)
    off = (length - lx) // 2
    y = np.empty(length, dtype=x.dtype)
    y[:off] = 0
    y[off:off+lx] = x
    y[off+lx:] = 0

    return y, off


def shrink_centered(x, length):
    """Shrinks an array at both ends.
    
    PARAMETERS
    ----------
    x: array_like (1d)
        Array to shrink.

    length: int
        Length of the target array.

    RETURNS
    -------
    y: array_1d
        Shrinked array.

    """
    x = np.asarray(x)
    lx = len(x)
    off0 = (lx - length) // 2
    off1 = - (off0 + (lx - length) % 2)
    y = x[off0:off1]

    return y


def plot_dataset_to(dataset, path, dataset_pos=None, xlim=None, ylim=None, verbose=True):
    if dataset.ndim != 2:
        raise ValueError("'dataset' must be a 2d array")

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    for igl, gl in enumerate(dataset):
        if verbose:
            print("Plotting plot %04d..." % igl, flush=True, end="\r", dest='terminal')
        ax.clear()
        ax.plot(gl)
        if dataset_pos is not None:
            p0, p1 = dataset_pos[igl]
        else:
            p0, p1 = 0, dataset.shape[1]
        ax.axvline(p0, c='red', ls='--')
        ax.axvline(p1, c='red', ls='--')
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)
        filename = os.path.join(path, f"dataset_plot-{igl:0{int(np.log10(len(dataset)))+1}d}.png")
        fig.savefig(filename)
    plt.close(fig)


def confused_plot(ax, cmat, labels, ilabels):
    ax.imshow(cmat, cmap=plt.get_cmap('Blues'), vmin=0, vmax=cmat.sum()*0.2)
    for i, j in itertools.product(ilabels, repeat=2):
        ax.annotate(str(int(cmat[i,j])), xy=(j,i), ha='center', va='center')
    ax.set_xticks(ilabels)
    ax.set_yticks(ilabels[:-1])  # Extra tick for 2nd-level label
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels[:-1])
    ax.set_xlim([-0.5, len(labels)-0.5])
    ax.set_ylim([-0.5, len(labels)-1.5])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.grid(False)