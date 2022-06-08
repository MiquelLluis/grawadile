import os
import time

import numpy as np
import scipy as sp
import scipy.optimize
from sklearn.decomposition import MiniBatchDictionaryLearning
import spams

from . import estimators
from . import patches_1d


os.environ['KMP_WARNINGS'] = 'FALSE'  # Remove temporal warning from OpenMP


# TODO: Actualitzar cridada a extract_patches_1d (canviat de nou a F-contiguous).
class DictionarySklearn(MiniBatchDictionaryLearning):
    """Basic Mini-Batch Dictionary Learning's interface for waveforms.

    Set of utilities to train waveform dictionaries using the MBDL methods from
    Scikit-learn package.

    Parameters
    ----------
    dict_init : array(n_atoms, n_features), tuple/list(array, array-like),
                MBDL(), str, or dict
        Source for the initial dictionary. If array, it is assumed to be the
        atoms of the initial dictionary and will be imported as is. If tuple or
        list, it must contain an array with signals (from where to extract the
        atoms) and an array-like with their positions along the lowest axis in
        case of being zero-padded. If str, it must be a valid file path to a
        saved dictionary. Other formats available are a Sklearn's MBDL
        instance, and a dict() with all the necessary attributes of an already
        created DictionarySklearn (or MBDL) instance.

    n_components : int, optional*
        Number of atoms.

    l_components : int, optional*
        Length of atoms.

    alpha : float, 1 by default
        Sparsity controlling parameter of the learning.

    batch_size : int, 3 by default
        Number of samples in each mini-batch.

    fit_algorithm : str, {'lars', 'cd'}, 'lars' by default
        Algorithm used to solve the dictionary learning problem.
        See [1] for more details.

    identifier : str, optional
        A word or short note to identify the dictionary.

    l2_normed : bool, True by default
        If True, normalize atoms to their L2-Norm.

    n_iter : int, 1000 by default
        Total number of iterations to perform.

    n_train : int, optional
        Number of patches (components) used to train the dictionary. It is
        merely informative.

    patch_min : int, 0 by default
        Minimum number of non-zero samples to include in each atom.

    random_state : int, 0 by default
        Seed used for random sampling.

    transform_algorithm : str, {‘lasso_lars’, ‘lasso_cd’, ‘lars’, ‘omp’,
                          ‘threshold’}, 'lasso_lars' by default
        Algorithm used to transform the data.
        See [1] for more details.

    transform_alpha : float, 1 by default
        If algorithm=’lasso_lars’ or algorithm=’lasso_cd’, alpha is the
        penalty applied to the L1 norm.
        See [1] for more details (where 'transform_alpha' is
        'transform_alpha').

    Attributes
    ----------
    components_ : array, [n_components, n_features]
        components extracted from the data

    inner_stats_ : tuple of (A, B) ndarrays
        Internal sufficient statistics that are kept by the algorithm.
        See [1] for more details.

    n_iter_ : int
        Number of iterations run.

    Notes
    -----
    * Must be provided if creating a new dictionary from a set of signals.

    References:

        [1]: Scikit-learn's MiniBatchDictionaryLearning class,
        (http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchDictionaryLearning.html),
        last accessed April 2018.

    """
    def __init__(self, dict_init=None, n_components=None, l_components=None,
                 patch_min=0, l2_normed=True, n_train=None, identifier='',
                 transform_algorithm='lasso_lars', random_state=0, **kwargs):
        # Import an already generated dictionary from a file
        if isinstance(dict_init, str):
            if dict_init.endswith(('.gz', '.gzip')):
                openf = gzip.open
            else:
                openf = open
            with openf(dict_init, 'rb') as f:
                self.__dict__.update(pickle.load(f))

        # Import an already generated dictionary from a dict()
        elif isinstance(dict_init, dict):
            self.__dict__.update(dict_init)

        # Import from an instance of MiniBatchDictionaryLearning.
        elif isinstance(dict_init, MiniBatchDictionaryLearning):
            l_components = dict_init.components_.shape[1]
            self._set_own_params(
                l_components, patch_min, l2_normed, identifier, n_train
            )
            self.__dict__.update(dict_init.__dict__)

        # Create a new dictionary
        elif isinstance(dict_init, (np.ndarray, tuple, list)):
            # optionally, get the initial components from a set of signals
            if isinstance(dict_init, (tuple, list)):
                collection, wave_pos = dict_init
                dict_init = patches_1d.extract_patches_1d(
                    collection, l_components, wave_pos, n_components,
                    l2_normed=l2_normed, patch_min=patch_min,
                    random_state=random_state
                )
            else:
                n_components, l_components = dict_init.shape
            self._set_own_params(
                l_components, patch_min, l2_normed, identifier, n_train
            )
            super().__init__(
                dict_init=dict_init, n_components=n_components,
                transform_algorithm=transform_algorithm,
                random_state=random_state, **kwargs
            )

        else:
            raise TypeError(
                f"'{type(dict_init).__name__}' is not recognized as any kind of dictoinary"
            )

    def __str__(self):
        n_train = "untrained" if self.n_train is None else f"{self.n_train:06d}"
        str_ = (
            f"dico_{self.identifier}_{self.alpha:.04f}_{self.n_components:04d}"
            f"_{self.l_components:03d}_{self.patch_min:03d}_{self.batch_size:03d}"
            f"_{n_train}_{self.n_iter:05d}"
        )
        return str_

    def fit(self, patches):
        """Train the dictionary with a set of patches.

        Calls `MiniBatchDictionaryLearning.fit`.

        Parameters
        ----------
        patches : array-like, shape (n_samples, n_features)
            Training vector.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        self.n_train = len(patches)
        super().fit(patches)
        return self

    def optimum_reconstruct(self, x0, x1, transform_alpha0, tol=1e-3, step=1,
                            method='SLSQP', full_out=False, wave_pos=None,
                            **kwargs_minimize):
        """Optimum reconstruction.

        Finds the optimum transform_alpha value which yields the reconstruction
        that minimizes the SSIM estimator between it and the original signal
        `x0`. It may take from seconds to hours.

        PARAMETERS
        ----------
        x0, x1 : array
            Original (normalized) and noisy signal.

        transform_alpha0 : float
            Initial guess of transform_alpha parameter.

        tol : float
            Tolerance for termination of the SciPy's minimize algorithm.
            1e-3 by default.

        step : int, optional
            Sample interval between each patch extracted from x. Determines
            the number of patches to be extracted. 1 by default.

        method : str, optional
            Method for solving the minimization problem, 'SLSQP' by default.
            For more details, see documentation page of
            "scipy.optimize.minimize".

        full_out : bool, optional
            If True, it also returns the OptimizedResult.
            False by default.

        wave_pos : array-like (p0, p1) of integers, optional
            Index positions of the signal where to compute the DSSIM. If not
            provided, it will be computed over all signals.

        **kwargs_minimize
            Additional keyword arguments passed to 'scipy.optimize.minimize'.

        RETURNS
        -------
        rec : ndarray
            Optimum reconstruction of the signal.

        res : OptimizedResult, only returned if `full_out == True`.
            Optimization result of SciPy's minimize function. Important
            attributes are: `x` the optimum transform_alpha and `fun` the
            optimum SSIM value. See 'scipy.optimize.OptimizeResult' for a
            general description of attributes.

        """
        # TODO: IF USED, THIS FUNCTION NEEDS TO BE UPDATED. SEE THE EQUIVALENT
        # FUNCTION FROM THE CLASS DictionarySpams.
        rec = [None]  # 'trick' for recovering the optimum reconstruction

        if wave_pos is None:
            def fun2min(transform_alpha):
                """Function to be minimized."""
                self.transform_alpha = transform_alpha
                rec[0] = self.reconstruct(x1, step=step)  # normalized
                return (1 - estimators.ssim(rec[0], x0)) / 2
        else:
            pos = slice(*wave_pos)

            def fun2min(transform_alpha):
                """Function to be minimized."""
                self.transform_alpha = transform_alpha
                rec[0] = self.reconstruct(x1, step=step)  # normalized
                return (1 - estimators.ssim(rec[0][pos], x0[pos])) / 2

        res = sp.optimize.minimize(
            fun2min,
            x0=transform_alpha0,
            method=method,
            tol=tol,
            **kwargs_minimize
        )
        rec = rec[0]

        return (rec, res) if full_out else rec

    def reconstruct(self, signal, step=1, norm=True, with_code=False):
        """Reconstruct a signal as a sparse combination of dictionary atoms.

        Parameters
        ----------
        signal : array
            Sample to be reconstructed.

        step : int, optional
            Sample interval between each patch extracted from signal.
            Determines the number of patches to be extracted. 1 by default.

        norm : boolean, optional
            Normalize the result to its maximum amplitude after adding the
            noise. True by default.

        with_code : boolean, optional.
            If True, also returns the coefficients array. False by default.

        Returns
        -------
        signal_rec : array
            Reconstructed signal.

        code : array, shape (n_samples, n_components)
            Transformed data, encoded as a sparse combination of atoms.

        """
        if signal.ndim != 2:
            raise ValueError("'signal' must be a 2d-array (column)")

        patches = patches_1d.extract_patches_1d(
            signal,
            patch_size=self.l_components,
            step=step
        )
        code = self.transform(patches)
        patches = np.dot(code, self.components_)
        signal_rec = patches_1d.reconstruct_from_patches_1d(patches, len(signal))

        if norm and signal_rec.any():
            coef = 1 / abs(signal_rec).max()
            signal_rec *= coef
            code *= coef

        return (signal_rec, code) if with_code else signal_rec

    def save(self, file, zipped=False):
        """Save the current state to a file."""
        data = self.__dict__
        if isinstance(file, str):
            if zipped:
                openf = gzip.open
            else:
                openf = open
            with openf(file, 'wb') as f:
                pickle.dump(data, f)
        else:
            pickle.dump(data, file)

    def _set_own_params(self, l_components, patch_min, l2_normed, identifier,
                        n_train):
        self.l_components = l_components
        self.patch_min = patch_min
        self.l2_normed = l2_normed
        self.identifier = identifier
        self.n_train = n_train


class DictionarySpams:
    """Mini-Batch Dictionary Learning interface for SPAMS-python.

    Set of utilities for dictionary learning and sparse encoding using the
    functions of SPAMS-python[1].

    Parameters
    ----------
    dict_init : 2d-array(p_size, d_size), optional
        Atoms of the initial dictionary.
        If None, 'signal_pool' must be given.

    signal_pool : 2d-array(samples, signals), optional
        Set of signals from where to randomly extract the atoms.
        Ignored if 'dict_init' is not None.

    wave_pos : 2d array-like (len(signals), 2), optional
        Position of each waveform inside 'signal_pool' from where to extract
        the atoms for the initial dictionary.
        If None, the whole array will be used.

    p_size : int, optional
        Atom length (patch size).
        If 'signal_pool' is not None, must be given.

    d_size : int, optional
        Number of atoms (dictionary size).
        If 'signal_pool' is not None, must be given.

    lambda1 : float, optional
        Regularization parameter of the learning algorithm.
        If None, will be requierd when calling 'train' method.

    batch_size : int, 64 by default
        Number of samples in each mini-batch.

    identifier : str, optional
        A word or note to identify the dictionary.

    l2_normed : bool, True by default
        If True, normalize atoms to their L2-Norm.

    n_iter : int, optional
        Total number of iterations to perform.
        If a negative number is provided it will perform the computation during
        the corresponding number of seconds. For instance n_iter=-5 learns the
        dictionary during 5 seconds.
        If None, will be required when calling 'train' method.

    n_train : int, optional
        Number of patches used to train the dictionary in case it has been
        trained already (just informative).

    patch_min : int, 1 by default
        Minimum number of samples within each 'wave_pos[i]' to include in each
        extracted atom when 'signal_pool' given.
        Will be ignored if 'wave_pos' is None.

    random_state : int, optional
        Seed used for random sampling.

    sc_lambda : float, optional
        Regularization parameter of the sparse coding transformation.

    trained : bool, False by default
        Flag indicating whether dict_init is an already trained dictionary.

    mode_traindl : int, 0 by default
        Refer to [1] for more information.

    mode_lasso : int, 2 by default
        Refer to [1] for more information.

    Attributes
    ----------
    dict_init : array(p_size, d_size)
        Atoms of the initial dictionary.

    components : array(p_size, d_size)
        Atoms of the current dictionary.

    n_iter : int
        Number of iterations performed in training.

    t_train : float
        Time spent training.

    identifier : str
        A word or note to identify the dictionary.

    References
    ----------
    [1]: SPAMS (for python), (http://spams-devel.gforge.inria.fr/), last
    accessed in october 2018.

    [2]: SciPy's Optimization tools, (https://docs.scipy.org/doc/scipy/reference/optimize.html),
    last accessed in February 2022.

    """
    def __init__(self, dict_init=None, signal_pool=None, wave_pos=None,
                 p_size=None, d_size=None, lambda1=None, batch_size=64,
                 identifier='', l2_normed=True, n_iter=None, n_train=None,
                 patch_min=1, random_state=None, sc_lambda=None, trained=False,
                 ignore_completeness=False, mode_traindl=0, mode_lasso=2):
        self.dict_init = dict_init
        self.components = dict_init
        self.wave_pos = wave_pos
        self.p_size = p_size
        self.d_size = d_size
        self.lambda1 = lambda1
        self.batch_size = batch_size
        self.identifier = identifier
        self.l2_normed = l2_normed
        self.n_iter = n_iter
        self.t_train = -n_iter if n_iter is not None and n_iter < 0 else None
        self.n_train = n_train
        self.patch_min = patch_min
        self.random_state = random_state
        self.sc_lambda = sc_lambda
        self.trained = trained
        self.ignore_completeness = ignore_completeness
        self.mode_traindl = mode_traindl
        self.mode_lasso = mode_lasso

        self._check_initial_parameters(signal_pool)

        # Explicit initial dictionary (trained or not).
        if self.dict_init is not None:
            self.p_size, self.d_size = self.dict_init.shape

        # Get the initial atoms from a set of signals.
        else:
            self.dict_init = patches_1d.extract_patches_1d(
                signal_pool,
                self.p_size,
                wave_pos=self.wave_pos,
                n_patches=self.d_size,
                l2_normed=self.l2_normed,
                patch_min=self.patch_min,
                random_state=self.random_state
            )
            self.components = self.dict_init

    def train(self, patches, lambda1=None, n_iter=None, verbose=False, **kwargs):
        """Train the dictionary with a set of patches.

        Calls 'spams.trainDL' to train the dictionary by solving the
        learning problem
            min_{D in C} (1/d_size) sum_{i=1}^d_size {
                (1/2)||x_i-Dalpha_i||_2^2  s.t. ||alpha_i||_1 <= lambda1
            } .

        Parameters
        ----------
        patches : 2d-array(samples, signals)
            Training patches.

        lambda1 : float, optional
            Regularization parameter of the learning algorithm.
            It is not needed if already specified at initialization.

        n_iter : int, optional
            Total number of iterations to perform.
            If a negative number is provided it will perform the computation
            during the corresponding number of seconds.
            It is not needed if already specified at initialization.

        verbose : bool, optional
            If True print the iterations (might not be shown in real time).

        **kwargs
            Passed directly to 'spams.trainDL', see [1].

        Additional parameters will be passed to the SPAMS training function.

        """
        if len(patches) != self.p_size:
            raise ValueError("the length of 'patches' must be the same as the"
                             " atoms of the dictionary")
        if n_iter is not None:
            self.n_iter = n_iter
        elif self.n_iter is None:
            raise TypeError("'n_iter' not specified")
            
        if lambda1 is not None:
            self.lambda1 = lambda1
        elif self.lambda1 is None:
            raise TypeError("'lambda1' not specified")

        self.n_train = patches.shape[1]

        tic = time.time()
        self.components, model = spams.trainDL(
            patches,
            D=self.dict_init,  # Cool-start
            batchsize=self.batch_size,
            lambda1=self.lambda1,
            iter=self.n_iter,
            mode=self.mode_traindl,  # Default mode is 2
            verbose=verbose,
            return_model=True,
            **kwargs
        )
        tac = time.time()

        self.trained = True

        if self.n_iter < 0:
            self.t_train = -self.n_iter
            self.n_iter = model['iter']
        else:
            self.t_train = tac - tic

    def reconstruct(self, signal, sc_lambda=None, step=1, l2_normed=True, with_code=False):
        """Reconstruct a signal as a sparse combination of dictionary atoms.

        Uses the 'lasso' function of SPAMS to solve the Lasso problem. By
        default it solves:
            min_{alpha} 0.5||x-Dalpha||_2^2 + lambda1||alpha||_1
                                        + 0.5 lambda2||alpha||_2^2

        Parameters
        ----------
        signal : ndarray
            Sample to be reconstructed.

        sc_lambda : float, optional
            Regularization parameter of the sparse coding transformation.
            It is not needed if already specified at initialization.

        step : int, 1 by default
            Sample interval between each patch extracted from signal.
            Determines the number of patches to be extracted. 1 by default.

        l2_normed : boolean, True by default
            Normalize the result so that the euclidian norm is 1.

        with_code : boolean, False by default.
            If True, also returns the coefficients array.

        Returns
        -------
        signal_rec : array
            Reconstructed signal.

        code : array(p_size, d_size), optional
            Transformed data, encoded as a sparse combination of atoms.
            Returned when 'with_code' is True.

        """
        if not isinstance(signal, np.ndarray):
            raise TypeError("'signal' must be a numpy array")
        
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)  # to column vector
            keepdims = False  # Return a 1d-array
        else:
            keepdims = True  # Return a 2d-array

        if sc_lambda is not None:
            self.sc_lambda = sc_lambda
        elif self.sc_lambda is None:
            raise TypeError("'sc_lambda' not specified")
        
        patches = patches_1d.extract_patches_1d(
            signal,
            patch_size=self.p_size,
            step=step
        )
        code = spams.lasso(
            patches,
            D=self.components,
            lambda1=self.sc_lambda,
            mode=self.mode_lasso
        )
        patches = self.components @ code

        signal_rec = patches_1d.reconstruct_from_patches_1d(patches, step, keepdims=keepdims)

        if l2_normed and signal_rec.any():
            norm = np.linalg.norm(signal_rec)
            signal_rec /= norm
            code /= norm

        return (signal_rec, code) if with_code else signal_rec

    def optimum_reconstruct(self, noisy, ref, sc_lambda0, loss_fun=None, l2_normed=True,
                            tol=1e-3, step=1, method='SLSQP', full_out=False,
                            wave_pos=None, verbose=False, **kwargs_minimize):
        """Optimum reconstruction according to a loss function.

        Finds the best reconstruction that can make the dictionary with its
        current parameters (previously configured). To do so, it looks for the
        optimum value of sc_lambda which minimizes the loss function
        'loss_fun'.

        The optimization of lambda is made in logarithmic scale.

        CAUTION: It might take seconds, hours, or even return 42.


        PARAMETERS
        ----------
        noisy, ref : 1d-array
            Reference (normalized) and noisy signal, respectively. They can be
            the same in case there is no 'ref' signal to compare with.

        sc_lambda0 : float
            Initial guess of sc_lambda parameter.

        loss_fun : function(rec, ref) -> float, optional
            Loss function which takes as argumetns 'rec' and 'ref', and returns
            a float value, which is the target to be minimized.
            If None, 'grawadile.estimators.dssim' will be used.

        l2_normed : bool, True by default
            Normalize the reconstructions so that their euclidian norm is 1.

        tol : float, 1e-3 by default
            Tolerance parameter of SciPy's 'minimize' function.

        step : int, 1 by default
            Sample interval between each patch extracted from noisy. Determines
            the number of patches to be extracted.

        method : str, 'SLSQP' by default
            Method for solving the minimization problem.
            See [1] for more details.

        full_out : bool, False by default
            If True, it also returns SciPy's OptimizedResult.

        wave_pos : array-like (p0, p1) of integers, optional
            Index positions of the signal where to compute the DSSIM.
            If None, DSSIM will be computed over the whole signals.

        verbose : bool, False by default
            If True, print to terminal each 'sc_lambda' tested by SciPy's
            'minimize' function.

        **kwargs_minimize
            Additional keyword arguments passed to SciPy's 'minimize' function.
            See [2] for more details.

        RETURNS
        -------
        rec : 1d-array
            Optimum reconstruction of the signal.

        res : OptimizedResult, returned if 'full_out' is True.
            Optimization result of SciPy's minimize function. Important
            attributes are: `x` the optimum log10(sc_lambda) and `fun` the
            optimum SSIM value.
            See [2] for a detailed description of attributes.

        """        
        if loss_fun is None:
            loss_fun = estimators.dssim

        sc_lambda_old = self.sc_lambda
        rec = None  # Will store the last reconstruction performed

        if wave_pos is None:
            def _fun2min(sc_lambda):
                """Function to be minimized."""
                nonlocal rec
                # Minimize in logarithmic scale for performance reasons.
                self.sc_lambda = 10 ** float(sc_lambda)  # in case a 1d-array given
                if verbose:
                    os.write(1, f"{self.sc_lambda}\n".encode())  # In case of using jupyterlab
                rec = self.reconstruct(noisy, step=step, l2_normed=l2_normed)
                return loss_fun(rec, ref)
        else:
            wave_pos = slice(*wave_pos)
            def _fun2min(sc_lambda):
                """Function to be minimized."""
                nonlocal rec
                # Minimize in logarithmic scale for performance reasons.
                self.sc_lambda = 10 ** float(sc_lambda)  # in case a 1d-array given
                if verbose:
                    os.write(1, f"{self.sc_lambda}\n".encode())  # In case of using jupyterlab
                rec = self.reconstruct(noisy, step=step, l2_normed=l2_normed)
                return loss_fun(rec[wave_pos], ref[wave_pos])

        res = sp.optimize.minimize(
            _fun2min,
            ref=np.log10(sc_lambda0),
            method=method,
            tol=tol,
            **kwargs_minimize
        )

        self.sc_lambda = sc_lambda_old

        return (rec, res) if full_out else rec

    def _check_initial_parameters(self, signal_pool):
        # Explicit initial dictionary.
        if self.dict_init is not None:
            if not isinstance(self.dict_init, np.ndarray):
                raise TypeError(
                    f"'{type(self.dict_init).__name__}' is not a valid 'dict_init'"
                )
            if not self.dict_init.flags.f_contiguous:
                raise ValueError("'dict_init' must be a F-contiguous array")
            if (self.dict_init.shape[0] >= self.dict_init.shape[1]
                and not self.ignore_completeness):
                raise ValueError("the dictionary must be overcomplete (p_size < d_size)")
        
        # Signal pool from where to extract the initial dictionary.
        elif signal_pool is not None:
            if not isinstance(signal_pool, np.ndarray):
                raise TypeError(
                    f"'{type(signal_pool).__name__}' is not a valid 'signal_pool'"
                )
            if not signal_pool.flags.f_contiguous:
                raise ValueError("'signal_pool' must be a F-contiguous array")
            if None in (self.p_size, self.d_size):
                raise TypeError(
                    f"'p_size' and 'd_size' must be explicitly provided along 'signal_pool'"
                )
            if self.p_size >= self.d_size:
                raise ValueError("the dictionary must be overcomplete (p_size < d_size)")
        
        # None of the above.
        else:
            raise ValueError("either 'dict_init' or 'signal_pool' must be provided")