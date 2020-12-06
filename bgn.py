import copy
import pickle

import numpy as np

from . import config as cfg



class AligoGaussianNoise:
    """Generate simulated aLIGO non-white gaussian noise.

    Parameters
    ----------
    noise : array_like, AligoGaussianNoise(), or str, optional
        Alternative noise array already generated. If another AligoGaussianNoise
        instance is given, it will create a copy of the instance. If 'str', it
        must be a valid file path of an instance saved with the 'save' method.

    t : float, optional
        Duration of noise to be generated. Will be omitted if a noise array is
        provided. 1s by default.

    sf : int, optional
        Sample frequency of the signal.

    random_seed : int or 1-d array_like, optional
        Seed for numpy.random.RandomState

    Attributes
    ----------
    noise : array
        Raw noise array generated at inicialization.

    duration : float
        Length of the noise array in seconds.

    """
    _version = '2018.12.04.0'

    def __init__(self, noise=None, t=1, sf=cfg.SF, psd=None, random_seed=None):
        self.sf = sf
        self.random_seed = random_seed
        self._i_version = self._version  # same as current class
        # PSD data to use
        if isinstance(psd, np.ndarray) and psd.ndim == 2 and psd.shape[0] == 2:
            self._psd = psd
        elif psd is None:
            # LALSuite's aLIGO ZERO Detuning, High Power PSD (starting from 20Hz!)
            self._psd = np.loadtxt("data/PSDaLIGOZeroDetHighPower", unpack=True, skiprows=44)
            self._psd[1] = self._psd[1] ** 2
        else:
            raise ValueError("if provided, 'psd' must be a ndarray matrix (2xN)")
        # Generate noise
        if noise is None:
            length = int(t * sf)
            self.duration = length / sf
            if t > 0:
                self.noise = self._gen_noise(length)
        # Import noise from array
        elif isinstance(noise, np.ndarray):
            self.duration = len(noise) / sf
            self.noise = noise
        # Import from a previous instance
        elif isinstance(noise, type(self)):
            self._import_from_dict(copy.deepcopy(noise.__dict__))
        # Import from a previous instance saved to a file
        elif isinstance(noise, str):
            with open(noise, 'rb') as f:
                dict_ = pickle.load(f)
            self._import_from_dict(dict_)
        else:
            _type = type(noise).__name__
            raise TypeError(
                "'%s' is not recognized as any kind of noise instance" % _type
            )

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

    def psd(self, f, margins=np.inf):
        """Interpolates the PSD from values in '_psd'."""
        return np.interp(f, *self._psd, left=margins, right=margins)

    def amplitude(self, f):
        """Noise amplitude."""
        return np.sqrt(self.psd(f))

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
        sum_ = sum((abs(hf[k])**2 / self.psd(f[k]) for k in range(ns//2)))

        return np.sqrt(4 * at**2 * af * sum_)

    def _gen_noise(self, length):
        """Generate the noise array."""
        sf = self.sf
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        even = length % 2 == 0
        # Positive frequencies + 0
        n = length // 2
        if not even:
            n += 1
        f = np.arange(0, sf/2, sf/2/n)
        # Noise components of the positive and zero frequencies in Fourier space
        # weighted by aLIGO's PSD amplitude and the normal distribution.
        nf_coefs = np.sqrt(length * sf * self.psd(f, margins=0)) / 2
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
        # Inverse Fast Fourier Transform
        ns = np.fft.ifft(nf).real

        return ns

    def _import_from_dict(self, dict_):
        """For version control when importing from previous instances."""
        version = dict_.pop('_i_version', 'oldest')
        if version == 'oldest':
            dict_.pop('n_samp')
            self.duration = dict_.pop('t')
            self.noise = dict_.pop('_noise')
        self.__dict__.update(dict_)