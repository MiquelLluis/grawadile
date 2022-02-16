import numpy as np


def extract_patches_1d(signals, patch_size, wave_pos=None, n_patches=None,
                       random_state=0, step=1, l2_normed=False,
                       patch_min=16):
    l_signals, n_signals = signals.shape
    max_pps = int((l_signals - patch_size) / step + 1)  # Maximum patches per signal
    np.random.seed(random_state)

    if wave_pos is None:
        max_patches = max_pps * n_signals
    else:
        max_patches = 0
        for j0, j1 in wave_pos:
            j0 += patch_min - patch_size
            j1 -= patch_min
            if j0 < 0:
                j0 = 0
            if j1 + patch_size >= l_signals:
                j1 = l_signals - patch_size
            max_patches += int(np.ceil((j1-j0)/step))

    if n_patches is None:
        n_patches = max_patches
    elif n_patches > max_patches:
        raise ValueError(
            f"the keyword argument 'n_patches' ({n_patches}) exceeds"
            f" the maximum number of patches that can be extracted ({max_patches})."
        )
    
    patches = np.empty((patch_size, n_patches), order='F')

    # All possible patches without wave_pos
    if n_patches == max_patches and wave_pos is None:
        k = 0  # Raveled index of (j,i)
        for i in range(n_signals):
            for j in range(0, max_pps*step, step):
                patches[:,k] = signals[j:j+patch_size,i]
                k += 1
    # All possible patches with wave_pos
    elif n_patches == max_patches:
        k = 0
        for i in range(n_signals):
            j0, j1 = wave_pos[i]
            j0 += patch_min - patch_size
            j1 -= patch_min
            if j0 < 0:
                j0 = 0
            if j1 + patch_size >= l_signals:
                j1 = l_signals - patch_size
            for j in range(j0, j1, step):
                if np.all(signals[j:j+patch_size,i] == 0):
                    print(i, j)
                    raise Exception
                patches[:,k] = signals[j:j+patch_size,i]
                k += 1
    # Limited number of patches without wave_pos
    elif wave_pos is None:
        for k in range(n_patches):
            i = np.random.randint(0, n_signals)
            j = np.random.randint(0, l_signals-patch_size)
            patches[:,k] = signals[j:j+patch_size,i]
    # Limited number of patches with wave_pos
    else:
        for k in range(n_patches):
            i = np.random.randint(0, n_signals)
            j0, j1 = wave_pos[i]  # Window indices
            j0 += patch_min - patch_size  # Extra space before the beggining of wave
            j1 -= patch_min  # Ensure taking at least 'patch_min' points
            if j0 < 0:
                j0 = 0
            if j1 + patch_size >= l_signals:
                j1 = l_signals - patch_size
            j = np.random.randint(j0, j1)
            patches[:,k] = signals[j:j+patch_size,i]

    # Normalize each patch to its L2 norm
    if l2_normed:
        patches /= np.linalg.norm(patches, axis=0)

    return patches


# REARRANGED TO M[samples][features]
def reconstruct_from_patches_1d(patches, signal_len):
    n, m = patches.shape
    step = int(round((signal_len - m) / (n - 1)))
    y   = np.zeros(signal_len)
    aux = np.zeros(signal_len)
    for i in range(n):       
        y[i*step:i*step+m]   += patches[i]
        aux[i*step:i*step+m] += 1.0  # faster than np.ones(m)
    aux[i*step+m:] = 1  # remaining samples of 'aux'
    return np.nan_to_num(y/aux)


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
    """Find the (index of) the patch given the index of the bin.

    Find the index of the patch to which corresponds a given index of a
    bind 'ibin' of an array split into multiple patches of length 'plength'
    separated with 'step' bins, adding up to a total of 'nbin'.

    """
    if ibin > nbin - plength:
        ipatch = (nbin - plength) // step + 1
    else:
        ipatch = ibin // step

    return ipatch