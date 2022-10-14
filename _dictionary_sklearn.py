import numpy as np
import scipy as sp
import scipy.optimize
from sklearn.decomposition import MiniBatchDictionaryLearning

from . import estimators
from . import patches_1d


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
