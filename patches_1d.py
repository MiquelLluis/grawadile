import warnings

import numpy as np


def extract_patches_1d(signals, patch_size, wave_pos=None, n_patches=None,
                       random_state=None, step=1, l2_normed=False,
                       patch_min=1, allow_allzeros=True):
    # TODO DOC:
    #   allow_allzeros: when extracting random patches, if False and l2_normed=True,
    #       generate another random window position until the l2 norm is != 0.
    if signals.ndim != 2:
        raise ValueError("'signals' must be a 2d-array")

    rng = np.random.default_rng(random_state)
    l_signals, n_signals = signals.shape
    max_pps = (l_signals - patch_size) / step + 1  # Maximum patches per signal
    if not max_pps.is_integer() and wave_pos is None:
        warnings.warn(
            "'signals' cannot be fully divided into patches, the last"
            f" {(max_pps-1)*step % step:.0f} bins of each signal will be left out",
            RuntimeWarning
        )
    max_pps = int(max_pps)

    # Compute the maximum number of patches that can be extracted and the
    # limits from where to extract patches for each signal.
    if wave_pos is None:
        window_limits = [(0, l_signals-patch_size+1)] * n_signals
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

    # Extract all possible patches.
    if n_patches == max_patches:
        k = 0
        for i in range(n_signals):
            p0, p1 = window_limits[i]
            for j in range(p0, p1, step):
                patches[:,k] = signals[j:j+patch_size,i]
                k += 1
    # Extract a limited number of patches randomly selected.
    # <<<
    elif l2_normed and not allow_allzeros:
        for k in range(n_patches):
            i = rng.integers(0, n_signals)
            j = rng.integers(*window_limits[i])
            signal_aux = signals[j:j+patch_size,i]
            while not signal_aux.any():
                j = rng.integers(*window_limits[i])
                signal_aux = signals[j:j+patch_size,i]
            patches[:,k] = signal_aux
    else:
        for k in range(n_patches):
            i = rng.integers(0, n_signals)
            j = rng.integers(*window_limits[i])
            patches[:,k] = signals[j:j+patch_size,i]
    # >>>

    # Normalize each patch to its L2 norm
    if l2_normed:
        patches /= np.linalg.norm(patches, axis=0)

    return patches


def reconstruct_from_patches_1d(patches, step, keepdims=False):
    l_patches, n_patches = patches.shape
    total_len = (n_patches - 1) * step + l_patches
    
    reconstructed = np.zeros(total_len)
    normalizer = np.zeros_like(reconstructed)
    for i in range(n_patches):
        reconstructed[i*step:i*step+l_patches] += patches[:,i]
        normalizer[i*step:i*step+l_patches] += 1
    normalizer[i*step+l_patches:] = 1
    reconstructed /= normalizer

    return reconstructed if not keepdims else reconstructed.reshape(-1,1)


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