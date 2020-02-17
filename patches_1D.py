# patches_1D.py
# Developed by Alejandro Torres <alejandro.torres@uv.es>

import numpy as np

from config import *

# PENDING TO REARRANGE TO ROW-MAJOR ORDER
def slidingWindow(sequence ,winSize ,step=1):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable."""

    # Verify the inputs
    try:
        it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")

    # Pre-compute number of chunks to emit
    numOfChunks = ((len(sequence ) -winSize ) /step ) +1

    # Do the work
    for i in range(0 , numOfChunks *step ,step):
        yield int(i), sequence[i: i +winSize]



# MODIFIED -> To C contiguous NOTATION (M[samples][features])
def extract_patches_1d(A, patch_size, wave_pos=None, n_patches=-1,
                       random_state=0, step=1, l2_normed=False,
                       patch_min=PATCH_MIN):
    # wave_pos -> Beggining and ending indices of each wave in A; 
    #
    n,m = np.atleast_2d(A).shape
    max_patches_per_sample = int((m - patch_size ) / step + 1)
    max_patches = max_patches_per_sample * n
    if n_patches > max_patches:
        _txt = (
            "WARNING: The keyword argument 'n_patches' ({:d}) exceeds"
            " the maximum number of patches that can be extracted ({:d})."
            " The maximum number of patches will be extracted instead."
        )
        print(_txt.format(n_patches, max_patches))
        n_patches = -1
    
    # All possible patches
    if (n_patches < 0) or (n_patches == max_patches):
        col = 0
        if n == 1:
            D = np.zeros([max_patches_per_sample, patch_size])
            for j in range(0 , max_patches_per_sample*step, step):
                D[col]=A[j:j+patch_size]
                col +=1
        else:
            D = np.zeros([max_patches, patch_size])
            for i in range(n):
                for j in range(0, max_patches_per_sample*step, step):
                    D[col]=A[i,j:j+patch_size]
                    col +=1
    
    # Limited number of patches
    else:
        D = np.zeros([n_patches, patch_size])
        np.random.seed(random_state)
        if wave_pos is None:
            for k in range(n_patches):
                i = np.random.randint(0, n)
                j = np.random.randint(0, m-2)
                D[k] = A[i,j:j+patch_size]
        else:
            for k in range(n_patches):
                i = np.random.randint(0,n)  # Sample index
                l0, l1 = wave_pos[i]  # Window indices
                l0 += patch_min - patch_size  # Extra space before the beggining of wave
                l1 -= patch_min  # Ensure taking at least 'patch_min' points
                if l0 < 0:
                    l0 = 0
                if l1 + patch_size >= m:
                    l1 = m - patch_size
                j = np.random.randint(l0, l1)
                #
                D[k] = A[i,j:j+patch_size]

    # Normalize each patch to its L2 norm
    if l2_normed:
        D /= np.linalg.norm(D, axis=1, keepdims=True)

    return D    
    
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