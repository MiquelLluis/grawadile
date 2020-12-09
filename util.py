import os
import time

import numpy as np


def printc(*args, dest='both', sep=' ', end='\n', **kw_args):
    """Print messages to the console and/or to the notebook."""
    dest = dest.lower()
    
    if dest in ('terminal', 'both'):
        bstring = sep.join([repr(s) for s in args]) + end
        if not isinstance(bstring, bytes):
            bstring = bstring.encode()
        os.write(1, bstring)
    
    if dest == 'both':
        print(*args, sep=sep, end=end, **kw_args)


def register(msg):
    """Register a message into a predefined register file."""
    _file = 'register.txt'
    with open(_file, 'a') as f:
        f.write(time.strftime("%Y%m%d,%H%M%S ") + msg + '\n')


def clearprint():
    print(" "*80, end='\r')


def closest_pow2(x):
    """Returns the closest power of 2 to 'x'."""
    if not isinstance(x, int):
        raise TypeError("'x' must be integer")
    elif x <= 0:
        raise ValueError("'x' must be a positive number")
    
    return 2 ** int(round(np.log2(x)))


def next_pow2(x):
    """Returns the next closest power of 2 to 'x'."""
    if not isinstance(x, int):
        raise TypeError("'x' must be integer")
    elif x <= 0:
        raise ValueError("'x' must be a positive number")
    
    return 2 ** int(np.ceil(np.log2(x)))


def _mif(x):
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


def bin2patch(ibin, nbin, plength, step):
    if ibin > nbin - plength:
        ipatch = (nbin - plength) // step + 1
    else:
        ipatch = ibin // step

    return ipatch