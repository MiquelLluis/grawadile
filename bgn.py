import copy
import pickle

import numpy as np

from . import config as cfg


class NonwhiteGaussianNoise:
    """Generate simulated non-white gaussian noise.

    Parameters
    ----------
    duration : float, optional
        Duration of noise to be generated, in seconds. It may change after
        genetarting the noise, depending on its sample frequency.
        Will be omitted if 'noise' is not None.

    noise : array-like, NonwhiteGaussianNoise() or str; optional
        Alternative noise array already generated. If another
        instance is given, it will create a copy. If 'str', it must be a
        valid file path of an instance saved with the 'save' method.
        If no 'noise' is provided, or it is just a noise array, 'psd' must be
        specified.

    psd : array-like, optional
        Approximated Power Spectral Density of the non-white part of the noise.
        If not given, it is assumed that 'noise' is a Pickle file or another
        instance of this class containing the attributes of a previous instance,
        including 'psd'. Otherwise, it must allways be provided.

    sf : int, optional
        Sample frequency of the signal. Must be provided in case of
        generating a new noise array or an already generated noise array
        is provided.

    random_seed : int or 1-d array_like, optional
        Seed for numpy.random.RandomState.

    Attributes
    ----------
    noise : array
        Raw noise array generated at inicialization.

    duration : float
        Duration of the noise in seconds.

    

    """
    _version = '2020.12.07.0'

    def __init__(self, duration=None, noise=None, psd=None, sf=None, random_seed=None):
        # Attribute declaration
        self._i_version = self._version  # same as current class
        self.noise = noise
        self.psd = psd
        self.duration = duration
        self.sf = sf
        self.random_seed = random_seed
        
        # Case 1: Generating new noise
        if noise is None:
            # First check if all kwargs were given
            if not isinstance(duration, (int, float)):
                raise TypeError("'duration' must be an integer or float number")
            if not isinstance(psd, (list, tuple, np.ndarray)):
                raise TypeError("the power spectral density 'psd' must be an array-like")
            if not isinstance(sf, int):
                raise TypeError("the sampling frequency 'sf' must be an integer")

            self._gen_noise()
        
        # Case 2: Importing an already generated noise array
        elif isinstance(noise, np.ndarray):
            # First check if all kwargs were given
            if not isinstance(psd, (list, tuple, np.ndarray)):
                raise TypeError("the power spectral density 'psd' must be an array-like")
            if not isinstance(sf, int):
                raise TypeError("the sampling frequency 'sf' must be an integer")

            self.duration = len(noise) / sf

        # Case 3: Importing another instance of NonwhiteGaussianNoise
        elif isinstance(noise, type(self)):
            self._import_from_instance(noise)

        # Case 4: Importing from an instance saved with the "save" method
        elif isinstance(noise, str):
            self._import_from_file(noise)

        else:
            _type = type(noise).__name__
            raise TypeError(f"'{_type}' is not recognized as any kind of noise instance")

    def __getitem__(self, key):
        """Allows accessing the noise data by time indices (in milliseconds)."""
        if isinstance(key, int):
            return self.noise[int(key / 1000 * self.sf)]
        elif not isinstance(key, slice):
            return TypeError("list indices must be integers or slices")

        if key.start:
            start = int(key.start / 1000 * self.sf)
        else:
            start = None
        if key.stop:
            stop = int(key.stop / 1000 * self.sf)
        else:
            stop = None
        if key.step:
            step = int(key.step / 1000 * self.sf)
        else:
            step = None
        sl = slice(start, stop, step)

        return self.noise[sl]

    def __len__(self):
        """Length of the noise data in milliseconds."""
        return self.noise.shape[-1]

    def __repr__(self):
        args = (type(self).__name__, self.duration, self.sf, self.random_seed)

        return '%s(t=%s, sf=%s, random_seed=%s)' % args

    def add_to(self, x, snr=1, limsx=None, pos=0, sf=cfg.SF, norm=True):
        """Add the simulated noise to the signal 'x'.

        Parameters
        ----------
        x : array
            Signal array. Its length must be lower or equal to
            the length of the noise array.

        snr : int or float, optional
            Signal to Noise Ratio. Defaults to 1.

        limsx : array-like or None, optional
            Limits of 'x' where to calculate the SNR.

        pos : int, optional
            Index position in the noise array where to inject the signal.
            0 by default.

        sf : int, optional
            Sample frequency of the signal.

        norm : boolean, optional
            Normalize 'x' to its maximum value after adding the noise.
            True by default.

        Returns
        -------
        noisy : array
            Signal array with noise at the desired SNR.

        scale : float
            Coefficient used to rescale the signal.

        """
        n = x.shape[-1]
        if n > len(self.noise):
            raise ValueError("'x' is larger than the noise array")

        if limsx is None:
            limsx = (None, None)
        limsx = slice(*limsx)

        scale = snr / self.snr(x[limsx], at=1/sf)
        x_noisy = x * scale + self.noise[pos:pos+n]

        if norm:
            x_max = abs(x_noisy).max()
            x_noisy /= x_max
            scale /= x_max

        return (x_noisy, scale)

    def save(self, file):
        """Save the current state to a file."""
        data = self.__dict__
        if isinstance(file, str):
            with open(file, 'wb') as f:
                pickle.dump(data, f)
        else:
            pickle.dump(data, file)

    def rescale(self, x, snr=1, sf=cfg.SF):
        """Rescale the signal 'x' to the given snr with respect to the PSD.

        Parameters
        ----------
        x : array
            Signal array.

        snr : int or float, optional
            Signal to Noise Ratio. Defaults to 1.

        sf : int, optional
            Sample frequency of the signal.

        Returns
        -------
        x_new : float
            Rescaled signal.

        """
        factor = snr / self.snr(x, at=1/sf)
        x_new = x * factor

        return (x_new, factor)

    def compute_psd(self, f, margins=np.inf):
        """Interpolates the PSD."""
        return np.interp(f, *self.psd, left=margins, right=margins)

    def amplitude(self, f):
        """Noise amplitude."""
        return np.sqrt(self.compute_psd(f))

    def snr(self, x, at=cfg.ST):
        """Signal to Noise Ratio.

        Parameters
        ----------
        x : array
            Signal array.
        at : float
            Time step (sample time step).

        Returns
        -------
        snr : float
            Signal to Noise Ratio

        """
        ns = len(x)
        hf = np.fft.fft(x)
        f = np.fft.fftfreq(ns, d=at)
        af = f[1]
        # Only sum over positive frequencies
        sum_ = sum((abs(hf[k])**2 / self.compute_psd(f[k]) for k in range(ns//2)))

        return np.sqrt(4 * at**2 * af * sum_)

    def _gen_noise(self):
        """Generate the noise array."""
        length = int(self.duration * self.sf)
        even = length % 2 == 0
        np.random.seed(self.random_seed)
        
        # Positive frequencies + 0
        n = length // 2
        if not even:
            n += 1
        f = np.arange(0, self.sf/2, self.sf/2/n)
        
        # Noise components of the positive and zero frequencies in Fourier space
        # weighted by the PSD amplitude and the normal distribution.
        nf_coefs = np.sqrt(length * self.sf * self.compute_psd(f, margins=0)) / 2
        nf_coefs = (nf_coefs * np.random.normal(size=n)
                    + 1j * (nf_coefs * np.random.normal(size=n)))
        
        # Noise components are ordered as follows:
        #    nf[0]     zero frequency term
        #    nf[1:n]   positive-frequency terms
        #    nf[#1:]   negative-frequency terms, starting from the most negative
        #              #1: n when 'even'; n+1 when not 'even'
        nf = np.empty(length, dtype=complex)
        nf[:n] = nf_coefs
        if even:
            nf[n] = 0
            nf[n+1:] = nf_coefs[:0:-1].conjugate()  # Condition  ñ(-f) = ñ*(f)
        else:
            nf[n:] = nf_coefs[:0:-1].conjugate()
        
        # The final noise array
        self.noise = np.fft.ifft(nf).real
        self.duration = len(self.noise) / self.sf  # Actual final duration

    def _import_from_dict(self, dict_):
        """For VERSION RETROCOMPATIBILITY when importing from previous instances."""
        version = dict_.pop('_i_version', 'oldest')
        
        if version == 'oldest':
            dict_.pop('n_samp')
            self.duration = dict_.pop('t')
            self.noise = dict_.pop('_noise')
        
        self.__dict__.update(dict_)

    def _import_from_instance(self, instance):
        dict_ = copy.deepcopy(noise.__dict__)
        self._import_from_dict(dict_)

    def _import_from_file(self, file_path):
        with open(noise, 'rb') as f:
            dict_ = pickle.load(f)
        self._import_from_dict(dict_)