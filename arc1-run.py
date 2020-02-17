# %matplotlib notebook

# import datetime
import gzip
# import itertools
import json
# import os
import pickle
# import random
# import sys
# import tempfile
import textwrap
import time

# import gwpy.timeseries
# import matplotlib as mpl
# from matplotlib import pyplot as plt
import numpy as np
# import pandas as pd
import spams
import tqdm
import scipy as sp
# import scipy.signal
# import sklearn

import glitch_learning
import patches_1D as p1d

# CONSTANTS -------------------------------------------------------------------
HELIX = 0
KOIFISH = 1
WHISTLE = 2
WF = ("Helix", "Koi-fish", "Whistle")
NWF = 3
FS = 2**14
TS = 1 / FS
GL_LENGTH = 4 * FS
N_TRAIN = 40
N_TEST = 20

# PATHS AND FILES --------------------------------------------------------------

_path = 'data/O1/glitch_classification/'
_path_den = _path + 'denoising/'
_path_den_dicos = _path_den + 'dictionaries/'
_path_den_tests = _path_den + 'tests/'
_path_clas = _path + 'classification/'
_path_clas_dicos = _path_clas + 'dictionaries/'
_path_clas_tests = _path_clas + 'tests/'
_path_gl = 'data/O1/glitches/'
_f_index = _path + 'index.json'
_f_gdb = _path_gl + 'glitches_database.pkl'
_f_gdbs = _path + 'glitches_database_shuffled.pkl'
_f_set_train = _path + "set_training_40-helix-koifish-whistle.pkl.gz"
_f_set_test = _path + "set_testing_20-helix-koifish-whistle.pkl.gz"

# LOAD DATA --------------------------------------------------------------------

with open(_f_index) as f:
    index = json.load(f)
with open(_f_gdb, 'rb') as f:
    gdb = pickle.load(f)
with open(_f_gdbs, 'rb') as f:
    gdbs = pickle.load(f)
with gzip.open(_f_set_train, 'rb') as f:
    set_train, set_train_pos = pickle.load(f)
with gzip.open(_f_set_test, 'rb') as f:
    set_test, set_test_pos =  pickle.load(f)

# AUXILIAR FUNCTIONS -----------------------------------------------------------

def extract_patches(collection, n_patches, corder, climits, psize, patch_min, random_state=0, l2_normed=False):
    m = np.atleast_2d(collection).shape[1]
    patches = np.zeros([n_patches, psize])
    np.random.seed(random_state)

    for k in range(n_patches):
        igl = corder[k]  # pre-computed glitch index
        # Window limits from where to randomly choose
        l0, l1 = climits[igl]
        l0 += patch_min - psize  # Extra space before the beggining of wave
        l1 -= patch_min  # Ensure taking at least 'patch_min' points
        if l0 < 0:
            l0 = 0
        if l1 + psize >= m:
            l1 = m - psize
        # Window position
        j = np.random.randint(l0, l1)
        
        patches[k] = collection[igl,j:j+psize]

    # Normalize each patch to its L2 norm
    if l2_normed:
        patches /= np.linalg.norm(patches, axis=1, keepdims=True)

    return patches    

def _update_index():
    with open(_f_index, 'w') as f:
        json.dump(index, f, indent=4, sort_keys=True)

def reconstruct_omp(x, dico, nonzeros, norm=True):
    x_ = x.reshape(-1, 1)  # column vector
    code = spams.omp(x_, D=dico, L=nonzeros).todense()
    if code.any():
        clas = np.ravel(dico @ code)
        if norm:
            clas /= np.max(abs(clas))
    else:
        clas = np.zeros_like(x)
    
    return clas

# ------------------------------------------------------------------------------

def main():
    dicos_key = '02'
    path_current = _path_clas_dicos + dicos_key + '/'

    loss_fun_chunk = 512

    def loss_function(a, b):
        x = a - b
        nx = len(x)
        loss = 0
        for i in range(0, nx, loss_fun_chunk):
            loss += sp.stats.normaltest(x[i:i+loss_fun_chunk]).pvalue
        return -loss

    def recursive_reconstruct(dico, x, step=None, threshold=1e-2, maxiter=1000, full_out=False):
        """Get the final reconstruction by recursively reconstructing the residual
        signal from the previous reconstruction:
            rec[n] = reconstruct(residual[n-1])
            residual[n] = residual[n-1] - rec[n]
        
        """
        rec_total = np.zeros_like(x)
        residual = x.copy()
        residual_prev = np.zeros_like(residual)
        iter_ = 0
        while np.linalg.norm(residual - residual_prev) > threshold and iter_ < maxiter:
            rec_partial = dico.reconstruct(residual, step=step, norm=False)
            rec_total += rec_partial
            residual_prev = residual.copy()
            residual -= rec_partial
            iter_ += 1
        return (rec_total, residual) if full_out else rec_total

    def optimum_reconstruct(dico, signal, tol=1e-3, step=1, method=None, wave_pos=None, bounds=None):
        """Optimum reconstruction according to a loss function."""
        if wave_pos is not None:
            wave_pos = slice(*wave_pos)
        
        clean = None  # Modified by 'fun2min'. It will hold the final reconstruction.
        
        # The minimization is performed in logarithmic scale for performance
        # and precision reasons.
        fun2min_maxmin = GL_LENGTH // loss_fun_chunk  # (max - min) value of fun2min
        logl_min = bounds[0]
        logl_maxmin = bounds[1] - logl_min
        def fun2min(log_lambda):
            """Function to be minimized."""
            nonlocal clean
            
            log_lambda = float(log_lambda)  # in case a 1d-array given
            lambda_ = 10 ** log_lambda
            
            dico.sc_lambda = lambda_
            clean = recursive_reconstruct(
                dico,
                signal,
                step=step,
                threshold=1e-2
            )
            
            if not np.any(clean):
                # Too-high-lambda controlling condition. See doc above.
                return fun2min_maxmin / logl_maxmin * (log_lambda - logl_min)
            else:
                return loss_function(signal[wave_pos], clean[wave_pos])

        res = sp.optimize.minimize_scalar(
            fun2min,
            method=method,
            bounds=bounds,
            options={'xatol': tol}
        )

        return (clean, res)

    iwf = WHISTLE
    wf = 'whistle'
    lambda_learn = '0.1'
    step = 8
    method = 'Bounded'
    bounds = [-1, 3]  # In log10
    tolerance = 1e-5  # Specific of the chosen method (see docs)

    f_dico = path_current + f'dict_clas_{wf}.pkl.gz'
    f_dico_den = (_path_den_dicos
                  + index['denoising']['dictionaries'][wf][str(lambda_learn)]['filename'])
    with open(f_dico_den, 'rb') as f:
        dico_den = pickle.load(f)

    _pbar = tqdm.tqdm(total=N_TRAIN, ncols=70)
    _time0 = time.time()
    dico = np.zeros((GL_LENGTH, N_TRAIN), order='F')  # FORTRAN ORDER
    lambdas_rec = np.empty(N_TRAIN)

    for igl, gl in enumerate(set_train[iwf]):
        p0, p1 = set_train_pos[iwf,igl]
        cl, res = optimum_reconstruct(
            dico_den,
            gl / abs(gl).max(),
            method=method,
            bounds=bounds,
            tol=tolerance,
            step=step,
            wave_pos=[p0, p1]
        )
        if cl.any():
            dico[p0:p1,igl] = cl[p0:p1]
            lambdas_rec[igl] = 10 ** res.x
        else:
            print(f"The glitch igl={igl} could not be reconstructed.\n")
            dico[:,igl] = np.nan
            lambdas_rec[igl] = np.nan
            
        if (time.time() - _time0) > 120:
            with gzip.open(f_dico, 'wb') as f:
                pickle.dump(dico, f)
            _time0 = time.time()
        _pbar.update()

    with gzip.open(f_dico, 'wb') as f:
        pickle.dump(dico, f)

    index['classification']['dictionaries'][dicos_key][wf] = {
        'filename': f_dico.split('/')[-1],
        'lambda_learn': '0.01',
        'step': 8,
        'method': 'Bounded',
        'bounds': [-1, 3],
        'tolerance': 1e-5
    }
    _update_index()

if __name__ == '__main__':
    main()