import numpy as np


def extract_patches_1d(signals, patch_size, wave_pos=None, n_patches=None,
                       random_state=None, step=1, l2_normed=False,
                       patch_min=16):
    rng = np.random.default_rng(random_state)
    l_signals, n_signals = signals.shape
    max_pps = int((l_signals - patch_size) / step + 1)  # Maximum patches per signal

    # Compute the maximum number of patches that can be extracted and the
    # limits from where to extract patches for each signal.
    if wave_pos is None:
        window_limits = [(0, l_signals-patch_size)] * n_signals
        max_patches = max_pps * n_signals
    else:
        window_limits = []
        max_patches = 0
        for p0, p1 in wave_pos:
            p0 += patch_min - patch_size
            p1 -= patch_min
            if p0 < 0:
                p0 = 0
            if p1 + patch_size >= l_signals:
                p1 = l_signals - patch_size
            window_limits.append((p0, p1))
            max_patches += int(np.ceil((p1-p0)/step))

    if n_patches is None:
        n_patches = max_patches
    elif n_patches > max_patches:
        raise ValueError(
            f"the keyword argument 'n_patches' ({n_patches}) exceeds"
            f" the maximum number of patches that can be extracted ({max_patches})."
        )
    
    patches = np.empty((patch_size, n_patches), order='F')

    # All possible patches.
    if n_patches == max_patches:
        k = 0
        for i in range(n_signals):
            p0, p1 = window_limits[i]
            for j in range(p0, p1, step):
                patches[:,k] = signals[j:j+patch_size,i]
                k += 1
    # Limited number of patches randomly selected.
    else:
        for k in range(n_patches):
            i = rng.integers(0, n_signals)
            j = rng.integers(*window_limits[i])
            patches[:,k] = signals[j:j+patch_size,i]

    # Normalize each patch to its L2 norm
    if l2_normed:
        patches /= np.linalg.norm(patches, axis=0)

    return patches


def reconstruct_from_patches_1d(patches, signal_len):
    l_patches, n_patches = patches.shape
    step = int(round((signal_len - l_patches) / (n_patches - 1)))
    
    reconstructed = np.zeros(signal_len)
    normalizer = np.zeros(signal_len)
    for i in range(n_patches):       
        reconstructed[i*step:i*step+l_patches] += patches[:,i]
        normalizer[i*step:i*step+l_patches] += 1
    normalizer[i*step+l_patches:] = 1

    return reconstructed / normalizer


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