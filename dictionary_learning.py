import gzip
import os
import pickle

import numpy as np
import scipy as sp
import scipy.optimize
from sklearn.decomposition import MiniBatchDictionaryLearning
import spams

from . import estimators
from . import patches_1d


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
        See [1] for more details (where 'transform_alpha' is 'transform_alpha').

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
            _type = type(dict_init).__name__
            raise TypeError(
                "'%s' is not recognized as any kind of dictoinary" % _type
            )

    def __str__(self):
        """Most identificative data of the dictionary."""
        n_train = 'untrained' if self.n_train is None else '%06d' % self.n_train
        return 'dico_%s_%.04f_%04d_%03d_%03d_%03d_%s_%05d' % (
            self.identifier, self.alpha, self.n_components, self.l_components,
            self.patch_min, self.batch_size, n_train, self.n_iter
        )

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
        clean : ndarray
            Optimum reconstruction of the signal.

        res : OptimizedResult, only returned if `full_out == True`.
            Optimization result of SciPy's minimize function. Important
            attributes are: `x` the optimum transform_alpha and `fun` the
            optimum SSIM value. See 'scipy.optimize.OptimizeResult' for a
            general description of attributes.

        """
        # TODO: IF USED, THIS FUNCTION NEEDS TO BE UPDATED. SEE THE EQUIVALENT
        # FUNCTION FROM THE CLASS DictionarySpams.
        clean = [None]  # 'trick' for recovering the optimum reconstruction

        if wave_pos is None:
            def fun2min(transform_alpha):
                """Function to be minimized."""
                self.transform_alpha = transform_alpha
                clean[0] = self.reconstruct(x1, step=step)  # normalized
                return (1 - metrics.ssim(clean[0], x0)) / 2
        else:
            pos = slice(*wave_pos)

            def fun2min(transform_alpha):
                """Function to be minimized."""
                self.transform_alpha = transform_alpha
                clean[0] = self.reconstruct(x1, step=step)  # normalized
                return (1 - metrics.ssim(clean[0][pos], x0[pos])) / 2

        res = sp.optimize.minimize(
            fun2min,
            x0=transform_alpha0,
            method=method,
            tol=tol,
            **kwargs_minimize
        )
        clean = clean[0]

        return (clean, res) if full_out else clean

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
        patches = patches_1d.extract_patches_1d(
            signal,
            patch_size=self.l_components,
            step=step
        )
        code = self.transform(patches)
        patches = np.dot(code, self.components_)
        signal_rec = patches_1d.reconstruct_from_patches_1d(patches, len(signal))

        if norm and signal_rec.any():  # Avoids ZeroDivisionError
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
    dict_init : array-like(m, n),
                tuple/list[array-like(m, p), array-like(p, 2)],
                str
        Source for the initial dictionary, where
            m: is the signal size,
            n: is the number of signals.
        If 'm' and 'n' are not specified, dict_init is assumed to be the atoms
        of the initial dictionary, otherwise it must contain an array-like with
        signals from where to extract the atoms and an array-like with the
        [start, end] indices of the signals in pairs.
        If str, it must be a valid file path to a saved 'DictionarySpams'
        instance.

    m : int
        Atom length. Must be provided if creating a new dictionary from a set
        of signals.

    n : int
        Number of atoms to be generated. Must be provided if creating a new
        dictionary from a set of signals.

    lambda1 : float, 1 by default
        Regularization parameter of the learning algorithm. Formerly 'alpha'.

    batch_size : int, 3 by default
        Number of samples in each mini-batch.

    identifier : str, optional
        A word or short note to identify the dictionary.

    l2_normed : bool, True by default
        If True, normalize atoms to their L2-Norm.

    n_iter : int, optional
        Total number of iterations to perform.
        If a negative number is provided it will perform the computation during
        the corresponding number of seconds.
        If dict_init is not trained, this parameter can be given to the 'train'
        method instead.

    n_train : int, optional
        Number of patches (components) used to train the dictionary if
        'dict_init' is already trained (merely informative).

    patch_min : int, 0 by default
        Minimum number of non-zero samples to include in each atom.

    random_state : int, 0 by default
        Seed used for random sampling.

    sc_lambda : float, 1 by default
        Regularization parameter of the sparse coding. Formerly
        'transform_alhpa'.

    trained : bool, False by default
        Flag indicating whether dict_init is an already trained dictionary.

    Attributes
    ----------
    dict_init : array(m, n)
        Atoms of the initial dictionary.
    components : array(m, n)
        Atoms of the dictionary.

    References:

        [1]: SPAMS (for python), (http://spams-devel.gforge.inria.fr/),
            last accessed October 2018.

    """
    def __init__(self, dict_init, m=None, n=None, lambda1=1, batch_size=3,
                 identifier='', l2_normed=True, n_iter=None, n_train=None,
                 patch_min=0, random_state=0, sc_lambda=1, trained=False,
                 mode_traindl=0, mode_lasso=2):
        # Initialize variables, some could be overwritten below.
        self.lambda1 = lambda1
        self.batch_size = batch_size
        self.identifier = identifier
        self.l2_normed = l2_normed
        if n_iter is not None:
            self.n_iter = n_iter
        if n_train is not None:
            self.n_train = n_train
        self.patch_min = patch_min
        self.random_state = random_state
        self.sc_lambda = sc_lambda
        self.trained = trained
        self.mode_traindl = mode_traindl
        self.mode_lasso = mode_lasso

        # Import an already generated dictionary from a file.
        if isinstance(dict_init, str):
            if dict_init.endswith(('.gz', '.gzip')):
                openf = gzip.open
            else:
                openf = open
            with openf(dict_init, 'rb') as f:
                self.__dict__.update(pickle.load(f))

        # Generate the initial dictionary from the set of signals in
        # dict_init[0].
        elif None not in (m, n):
            collection, wave_pos = dict_init
            # TODO: new function to avoid having to compute the transposed.
            collection = collection.T
            self.dict_init = patches_1d.extract_patches_1d(
                collection, m, wave_pos, n, l2_normed=l2_normed, patch_min=patch_min,
                random_state=random_state
            )
            self.dict_init = self.dict_init.T  # This leaves dict_init as a Fortran array

        # Take dict_init as the initial dictionary.
        elif isinstance(dict_init, (list, tuple, np.ndarray)):
            dict_init = np.asarray(dict_init)
            m, n = dict_init.shape
            if m >= n:
                raise ValueError("the dictionary must be overcomplete (m < n).")
            if trained:
                self.components = dict_init
                self.trained = trained
            else:
                self.dict_init = dict_init
            pass

        # Raise an exception if none of the above
        else:
            _type = type(dict_init).__name__
            raise TypeError(
                "'%s' is not recognized as an instance of DictionarySpams" % _type
            )

    def train(self, patches, n_iter=None, **kwargs):
        """Train the dictionary with a set of patches.

        Calls 'spams.trainDL' to train the dictionary by solving the
        learning problem
            min_{D in C} (1/n) sum_{i=1}^n (1/2)||x_i-Dalpha_i||_2^2  s.t. ...
                                                     ||alpha_i||_1 <= lambda1 .

        Parameters
        ----------
        patches : array-like(n_samples, n_features)
            Training vector.

        n_iter : int, optional
            Total number of iterations to perform.
            If a negative number is provided it will perform the computation
            during the corresponding number of seconds.
            It is not needed if already specified at initialization.

        n_train : int, optional
            Number of patches (components) used to train the dictionary.
            It is not needed if already specified at initialization.

        Additional parameters will be passed to the SPAMS training function.

        """
        if len(patches) != len(self.dict_init):
            raise ValueError("the length of 'patches' must be the same as the"
                             " atoms of the dictionary.")
        if n_iter is None and self.n_iter is None:
            raise TypeError("'n_iter' not specified.")

        if n_iter is not None:
            self.n_iter = n_iter
        if 'lambda1' in kwargs:
            self.lambda1 = kwargs.pop('lambda1')
        self.n_train = patches.shape[1]

        self.components = spams.trainDL(
            patches,
            D=self.dict_init,
            batchsize=self.batch_size,
            lambda1=self.lambda1,
            iter=self.n_iter,
            mode=self.mode_traindl,  # default mode is 2
            **kwargs
        )
        self.trained = True

    def optimum_reconstruct(self, x0, x1, sc_lambda0, loss_fun=None,
                            tol=1e-3, step=1, method='SLSQP', full_out=False,
                            wave_pos=None, **kwargs_minimize):
        """Optimum reconstruction according to a loss function.

        Finds the best reconstruction that can make the dictionary with its
        current parameters (previously configured). To do so, it looks for the
        optimum value of sc_lambda which minimizes the loss function 'loss_fun'.

        The optimization of lambda is made in a logarithmic scale.

        CAUTION: It might take seconds, hours, or even return 42.


        PARAMETERS
        ----------
        x0, x1 : array
            Original (normalized) and noisy signal, respectively. They can be
            the same in case there is no "original" signal to compare with.

        sc_lambda0 : float
            Initial guess of sc_lambda parameter.

        loss_fun : function(x0, rec) -> float, optional
            Loss function which takes as argumetns 'x0' and 'rec', and returns
            a float value to be minimized. If none, the DSSIM will be used.

        tol : float, optional
            Tolerance for termination of the SciPy's minimize_scalar algorithm.
            1e-3 by default. (It is assigned to the corresponding solver chosen
            by 'method').

        step : int, optional
            Sample interval between each patch extracted from x. Determines
            the number of patches to be extracted. 1 by default.

        method : str, optional
            Method for solving the minimization problem, 'SLSQP' by default.
            For more details, see documentation page of
            "scipy.optimize.minimize_scalar".

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
        clean : ndarray
            Optimum reconstruction of the signal.

        res : OptimizedResult, only returned if `full_out == True`.
            Optimization result of SciPy's minimize function. Important
            attributes are: `x` the optimum log10(sc_lambda) and `fun` the
            optimum SSIM value. See 'scipy.optimize.OptimizeResult' for a
            general description of attributes.

        """        
        if loss_fun is None:
            loss_fun = metrics.dssim
            norm = True
        else:
            norm = False  # Lets loss_fun manage the scaling

        clean = [None]  # 'trick' for recovering the optimum reconstruction

        if wave_pos is not None:
            wave_pos = slice(*wave_pos)

        # The minimization is performed in logarithmic scale for performance
        # reasons.
        def fun2min(sc_lambda):
            """Function to be minimized."""
            self.sc_lambda = 10 ** float(sc_lambda)  # in case a 1d-array given
            os.write(1, "{}\n".format(self.sc_lambda).encode())
            clean[0] = self.reconstruct(x1, step=step, norm=norm)
            return loss_fun(x0[wave_pos], clean[0][wave_pos])

        res = sp.optimize.minimize(
            fun2min,
            x0=np.log10(sc_lambda0),
            method=method,
            tol=tol,
            **kwargs_minimize
        )
        clean = clean[0]

        return (clean, res) if full_out else clean

    def reconstruct(self, signal, step=1, norm=True, with_code=False, **kwargs):
        """Reconstruct a signal as a sparse combination of dictionary atoms.

        Uses the 'lasso' function of SPAMS to solve the Lasso problem. By
        default it solves:
            min_{alpha} 0.5||x-Dalpha||_2^2 + lambda1||alpha||_1
                                        + 0.5 lambda2||alpha||_2^2

        Parameters
        ----------
        signal : array-like
            Sample to be reconstructed.

        step : int, optional
            Sample interval between each patch extracted from signal.
            Determines the number of patches to be extracted. 1 by default.

        norm : boolean, optional
            Normalize the result to its maximum amplitude after adding the
            noise. True by default.

        with_code : boolean, optional.
            If True, also returns the coefficients array. False by default.

        Additional parameters will be passed to the SPAMS function 'lasso'.

        Returns
        -------
        signal_rec : array
            Reconstructed signal.

        code : array(m, n)
            Transformed data, encoded as a sparse combination of atoms.

        """
        # A 'lambda1' here is assumed to be the lambda for the reconstruction.
        if 'lambda1' in kwargs:
            self.sc_lambda = kwargs.pop('lambda1')
        elif 'sc_lambda' in kwargs:
            self.sc_lambda = kwargs.pop('sc_lambda')
        signal = np.asarray(signal)
        # TODO: new function to avoid having to 'compute' the transposed
        patches = patches_1d.extract_patches_1d(
            signal,
            patch_size=len(self.components),
            step=step
        ).T
        code = spams.lasso(
            patches,
            D=self.components,
            lambda1=self.sc_lambda,
            mode=self.mode_lasso
        ).todense()
        patches = np.dot(self.components, code)
        # TODO: new function to avoid having to compute the transposed
        signal_rec = patches_1d.reconstruct_from_patches_1d(
            np.ascontiguousarray(patches.T),  # (p, m) with C-contiguous order
            len(signal)
        )

        if norm and signal_rec.any():  # Avoids ZeroDivisionError
            coef = 1 / abs(signal_rec).max()
            signal_rec *= coef
            code *= coef

        return (signal_rec, code) if with_code else signal_rec