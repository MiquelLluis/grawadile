#!/usr/bin/env python
"""
all_in_one_optimization.py

Run all tests from the Glitch Classification O1 Extended notebook to
optimize all the hyper-parameters of the classification algorithm.


Author:  Miquel Lluís Llorens Monteagudo <millomon@alumni.uv.es>


TODO
----
- Finish check over all the module (lost changes last friday)
- Finish step 1

"""
import copy
import datetime
import gzip
import itertools
import json
import logging
import os
import pathlib
import pdb  # DEBUG
import pickle
import random
import sys
import tempfile
import textwrap
import time

import numpy as np
import pandas as pd
import spams
import scipy as sp
import scipy.signal
import sklearn

import glitch_learning
import patches_1D as p1d


###############################################################################
#
# CONSTANTS AND CONFIGURATIONS
#
###############################################################################

KOIFISH = KF = 1
WHISTLE = WI = 2
LOST = 3
WF = {KOIFISH: "koifish", WHISTLE: "whistle"}
NWF = 2
FS = 2**14  # Hz
GL_LENGTH = 4 * FS
N_KOIFISH = 950  # initially 1000, but some had to be removed due to other noise sources
N_WHISTLE = 230  # initially 320, '' '' '' 
N_KOIFISH_TRAIN = 570  # 60% (initially 800)
N_WHISTLE_TRAIN = 136  # 60%
N_KOIFISH_CVT = 190  # CV and Test: 20% each
N_WHISTLE_CVT = 46  # CV and Test: 20% each

# Initial hyper-parameters:
HYPER_PARAMETERS_BASE = {
    # Trained dictionaries
    'l_atoms_den': None,
    'n_atoms_den': None,
    'lambda_learn': {KF: 0.005, WI: 0.1},
    'iters_learn': 100_000,
    'n_patches_learn': 100_000,
    'batch_size': 4,
    # Untrained dictionaries
    'l_atoms_clas_frac': None,  # TODO: With respect to what was it?
    'l_atoms_clas': GL_LENGTH,
    'n_atoms_clas_frac': None,  # TODO: With respect to what was it?
    'n_atoms_clas': {KF: N_KOIFISH_CVT, WI: N_WHISTLE_CVT},
    # Other
    'patch_min': None,  # Minimum samples of a signal to include in a patch
    'patch_min_frac': 0.5,  # Fraction in terms of the atom's length.
    'random_state': 42
}
# All the hyper-parameters of each step. Some of them will be modified (in a
# copy) as they are optimized by the step functions.
HYPER_PARAMETERS = (
    # Step 1
    {
        **HYPER_PARAMETERS_BASE,
        # To be optimized:
        'l_atoms_den': (64, 128, 256, 512, 1024),
        'n_atoms_den': (256, 512, 1024, 2048, 4096),
        # Other
        'step': 8
    },
    # # Step 2
    # {
    #     **HYPER_PARAMETERS_BASE,
    #     # To be optimized:
    #     'lambda_learn': {KF: 0.005, WI: 0.1}
    # }
)

PATH_O1 = pathlib.Path('data/O1/')
PATH_GL = PATH_O1 / 'glitches'

PATH_MAIN = PATH_O1 / 'glitch_classification_simple_whitening_ampliated'
PATH_DEN = PATH_MAIN / 'denoising'
PATH_DEN_DICOS = PATH_DEN / 'dictionaries'
PATH_DEN_TESTS = PATH_DEN / 'tests'
PATH_CLAS = PATH_MAIN / 'classification'
PATH_CLAS_DICOS = PATH_CLAS / 'dictionaries'
PATH_CLAS_TESTS = PATH_CLAS / 'tests'

F_INDEX = PATH_MAIN / 'index.json'
F_GDB = PATH_GL / 'glitches_database_all.pkl'
F_GDBS = PATH_MAIN / 'glitches_database_shuffled_all.pkl'
F_SET = PATH_GL / "whitened_set_all_950,230-koifish-whistle_4length.pkl"
F_SAVED_TMP_PROGRESS = PATH_MAIN / 'saved_tmp_progress.pkl'


###############################################################################
#
# AUXILIAR FUNCTIONS
#
###############################################################################

def cost_function_recursive_gaussianity(dico, x, bounds, wave_pos=None, verbose=False):
    """Returns a cost function in terms of the regularization parameter.

    Returns the cost function of the recursive reconstruction with respect to
    the original signal 'x', performed by a dictionary 'dico', in terms of its
    regularization parameter 'lambda_'.
    It also returns a list 'clean' which will hold the last recursive
    reconstruction performed by 'cost_function', useful after a minimization
    process in order to avoid having to reconstruct again the original signal
    with the optimized 'lambda_'.

    From an unknown value of 'lambda_' the dictionary will only produce zeros,
    which makes the loss function constant. In order to avoid issues with some
    minimization algorithms, in this case it does a linear extrapolation using
    the value of the loss function as its 'right-most' value.

    """
    if wave_pos is None:
        wave_pos = slice(None)

    # Components of the extrapolation:
    x_section = x[wave_pos]
    cost_zero_clean = loss_function_gaussianity(x_section, np.zeros_like(x_section))
    coef_a = cost_zero_clean / (bounds[1] - bounds[0])
    coef_b = -bounds[0] * coef_a

    clean = [None]  # Modified by 'cost'. It will hold the final reconstruction.

    def cost_function(lambda_):
        lambda_ = float(lambda_)  # in case a 1d-array given
        dico.sc_lambda = lambda_

        if verbose:
            print_logg(f"Cost function evaluating {lambda_} ...")

        clean_ = recursive_reconstruct(dico, x, **kwargs_recursive_reconstruct)
        clean[0] = clean_
        
        if np.any(clean_):
            result = loss_function_gaussianity(x_section, clean_[wave_pos])
        else:
            # Too-high-lambda controlling condition. See doc above.
            result = lambda_ * coef_a + coef_b
            
        return result

    return cost_function, clean


def crop_center_1d(x, length, copy=False, axis=-1):
    """Returns 'x' cropped to the final 'length' along 'axis'."""
    x = np.asarray(x)
    
    p0 = (x.shape[axis] - length) // 2
    p1 = p0 + length
    slc = [slice(None)] * len(x.shape)
    slc[axis] = slice(p0, p1)
    slc = tuple(slc)

    return x[slc] if not copy else x[slc].copy()  


def load_json(file):
    with open(file) as f:
        json_object = json.load(f)

    return json_object


def load_temporal_progress(file):
    """
    Loads all objects stored at 'file' as a dictionary.

    """
    with open(file, 'rb') as f:
        objects = pickle.load(f)

    return objects


def loss_function_gaussianity(original, reconstructed, patch_size=512):
    """Loss function between two signals, based on the normality test.

    Loss function defined to measure how well the reconstructed glitch in
    'reconstructed' takes out the original glitch embeded in noise in
    'original'.

    This is done by performing the normality test of the residual between the
    original signal 'original' and the reconstructed glitch 'reconstructed'.
    The more gaussian, the better the glitch was reconstructed.

    The more long the array is, the smaller the P-value will be, because the
    component of the glitch will also be smaller compared to the whitened
    noise. Therefore we split the residual (a - reconstructed) into small
    patches, compute the normality test over these ones and add up all the
    P-values. After some tests, a reasonable value seemed to be 512 samples.

    PARAMETERS
    ----------
    original, reconstructed: array-like
        Original signal and reconstructed glitch, respectively. Must be of the
        same length.
    patch_size: int
        Size of the patches into which to split up the residual. The normality
        test will be computed over these patches.

    RESULTS
    -------
    loss: float
        Computed loss between 'original' and 'reconstructed'. The lower, the
        more gaussian the residual is, the better 'reconstructed' was
        reconstructed.
        Range: from 0 to -nx/patch_size (i.e. the number of patches).

    """
    x = original - reconstructed
    nx = len(x)
    loss = 0
    for i in range(0, nx, patch_size):
        loss += sp.stats.normaltest(x[i:i+patch_size]).pvalue
    loss *= -1

    return loss  # Maximize P-Value


def print_logg(msg, level='info'):
    """Print and send to logging.info a message 'msg'."""
    print(msg)
    if level == 'info':
        logging.info(msg)
    else:
        raise ValueError("Invalid logging level.")


def random_save(obj, prefix=None, suffix=None, dir=None, compress=False, **kwargs_tempfile):
    """Save an object to a file with a random name.

    Save 'obj' using Pickle to a (permanent) file named with random characters
    using tempfile.NamedTemporaryFile.

    PARAMETERS
    ----------
    obj: object
        Object to be pickled.
    prefix, suffix: str, optionals
        Prefix and suffix for the random name of the file.
    dir: str, optional
        Directory where to save the object. If None, a default
        (platform-dependent) directory is used.
    compress: bool
        If True, uses gzip to compress the data.

    RETURNS
    -------
    filename: str
        Name of the file.

    """
    with tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, dir=dir, delete=False,
            **kwargs_tempfile) as file:
        if compress:
            with gzip.open(file, 'wb') as gfile:
                pickle.dump(obj, gfile)
        else:
            pickle.dump(obj, file)
    filename = os.path.basename(file.name)

    return filename


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


def recursive_reconstruct(dico, x, step=None, threshold=None, maxiter=1000, full_out=False):
    """
    Get the final reconstruction by recursively reconstructing the residual
    signal from the previous reconstruction:

        rec[n] = reconstruct(residual[n-1])
        residual[n] = residual[n-1] - rec[n]

    Keyword argumetns with default to 'None' need to be specified.
    
    """
    if None in (step, threshold):
        raise ValueError("Keyword arguments which default to 'None' need to be specified.")

    reconstruction = np.zeros_like(x)
    residual = x.copy()
    iterations = 0
    keep_going = True

    while keep_going:
        partial = dico.reconstruct(residual, step=step, norm=False)
        reconstruction += partial
        residual_prev = residual.copy()
        residual -= partial
        iterations += 1

        keep_going = (partial.any()
            and iterations < maxiter
            and np.linalg.norm(residual - residual_prev) > threshold)
    
    return (reconstruction, residual) if full_out else reconstruction


def save_results(results):
    pass


def save_temporal_progress(file, **objects):
    """
    Saves all objects passed as keyword arguments to a dictionary in a temporal
    file, which will be overwritten at every call.

    """
    with open(file, 'wb') as f:
        pickle.dump(objects, f)


def save_json(file, object_):
    with open(file, 'w') as f:
        json.dump(object_, f, indent=2, sort_keys=True)


###############################################################################
#
# Optimization steps and Main
#
###############################################################################

def optimization_step_1(set_train, set_train_pos, set_cv, set_cv_pos,
        n_atoms_den=None, l_atoms_den=None, lambda_learn=None, step=None, iters_learn=None,
        n_patches_learn=None, patch_min_frac=None, batch_size=None, random_state=None):
    """
    Double optimization of the number and lenght of the atoms of the learned
    dictionaries.

    """
    logging.info("Executing optimization step 1.")

    ln_grid = []
    for l_atoms, n_atoms in itertools.product(l_atoms_den, n_atoms_den):
        if l_atoms < n_atoms:
            ln_grid.append([l_atoms, n_atoms])
    n_atoms_max = max(n_atoms_den)

    #
    # Generate the dictionaries.
    #
    
    dicos = np.empty((NWF, len(ln_grid)), dtype=object)
    l_atoms_prev = None
    time0 = time.time()

    for [iwf, kwf], [iln, ln] in itertools.product(enumerate(WF), enumerate(ln_grid)):
        l_atoms, n_atoms = ln

        print_logg(f"Training {WF[kwf]} dico {l_atoms} x {n_atoms}")

        # Generate the initial dictionary and the set of training patches.
        if l_atoms != l_atoms_prev:
            patch_min = int(l_atoms * patch_min_frac)
            initial_d = p1d.extract_patches_1d(
                set_train[kwf],
                l_atoms,
                n_patches=n_atoms_max,
                wave_pos=set_train_pos[kwf],
                l2_normed=True,
                patch_min=patch_min,
                random_state=random_state
            ).T  # [length, atoms]
            training_patches = p1d.extract_patches_1d(
                set_train[kwf],
                l_atoms,
                wave_pos=set_train_pos[kwf],
                n_patches=n_patches_learn,
                l2_normed=True,
                patch_min=patch_min,
                random_state=random_state
            ).T
            l_atoms_prev = l_atoms
        
        dico = glitch_learning.GlitchDictSpams(
            initial_d[:,:n_atoms],
            lambda1=lambda_learn[kwf],
            batch_size=batch_size,
            patch_min=patch_min,
            identifier=WF[kwf],
            random_state=random_state
        )

        dico.train(training_patches, n_iter=iters_learn)

        dicos[iwf,iln] = dico
        save_temporal_progress(F_SAVED_TMP_PROGRESS, dicos=dicos)

    print_logg(f"Training completed in {datetime.timedelta(seconds=time.time()-time0)}")

    #
    # Denoising reconstructions: optimize lambda_rec and get the p-values.
    #

    p_values = np.empty((len(ln_grid), NWF), dtype=object)
    p_values.fill([])  # Empty lists because the number of glitches is unbalanced

    # for [iln, ln], [iwf, kwf] in itertools.product(enumerate(ln_grid), enumerate(WF)):
    #     for [igl, glitch] in enumerate(set_cv[kwf]):
            


def optimization_step_2():
    pass


def optimization_step_3():
    pass


def optimization_step_4():
    pass


def optimization_step_5():
    pass


def optimization_step_6():
    pass




def main():

    logging.basicConfig(
        filename=PATH_MAIN/"all_in_one_optimization.log",
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG
    )
    print_logg("#"*79)
    print_logg("ALL IN ONE OPTIMIZATION.py")
    
    # LOAD DATA

    with open(F_INDEX) as f:
        index = json.load(f)
    with open(F_GDB, 'rb') as f:
        gdb = pickle.load(f)
    with open(F_GDBS, 'rb') as f:
        gdbs = pickle.load(f)
    with open(F_SET, 'rb') as f:
        set_all, set_all_pos = pickle.load(f)

    set_train = {
        KOIFISH: set_all[KOIFISH][:N_KOIFISH_TRAIN],
        WHISTLE: set_all[WHISTLE][:N_WHISTLE_TRAIN]
    }
    set_cv = {
        KOIFISH: set_all[KOIFISH][N_KOIFISH_TRAIN:N_KOIFISH_TRAIN+N_KOIFISH_CVT],
        WHISTLE: set_all[WHISTLE][N_WHISTLE_TRAIN:N_WHISTLE_TRAIN+N_WHISTLE_CVT]
    }
    set_test = {
        KOIFISH: set_all[KOIFISH][N_KOIFISH_TRAIN+N_KOIFISH_CVT:N_KOIFISH],
        WHISTLE: set_all[WHISTLE][N_WHISTLE_TRAIN+N_WHISTLE_CVT:N_WHISTLE]
    }
    set_train_pos = {
        KOIFISH: set_all_pos[KOIFISH][:N_KOIFISH_TRAIN],
        WHISTLE: set_all_pos[WHISTLE][:N_WHISTLE_TRAIN]
    }
    set_cv_pos = {
        KOIFISH: set_all_pos[KOIFISH][N_KOIFISH_TRAIN:N_KOIFISH_TRAIN+N_KOIFISH_CVT],
        WHISTLE: set_all_pos[WHISTLE][N_WHISTLE_TRAIN:N_WHISTLE_TRAIN+N_WHISTLE_CVT]
    }
    set_test_pos = {
        KOIFISH: set_all_pos[KOIFISH][N_KOIFISH_TRAIN+N_KOIFISH_CVT:N_KOIFISH],
        WHISTLE: set_all_pos[WHISTLE][N_WHISTLE_TRAIN+N_WHISTLE_CVT:N_WHISTLE]
    }

    # Run optimization steps

    optimization_steps = [
        optimization_step_1
        # optimization_step_2,
        # optimization_step_3,
        # optimization_step_4,
        # optimization_step_5,
        # optimization_step_6
    ]
    hyper_parameters = [parameters.copy() for parameters in HYPER_PARAMETERS]
    results_steps = []

    for optimization_step, hyper_parameters_step in zip(optimization_steps, hyper_parameters):
        results = optimization_step(
            set_train,
            set_train_pos,
            set_cv,
            set_cv_pos,
            **hyper_parameters_step
        )
        results_steps.append(results)
        save_results(results_steps)




if __name__ == '__main__':
    main()