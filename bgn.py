import copy
import pickle

import numpy as np

from . import config as cfg


class NonwhiteGaussianNoise:
    """Simulate non-white gaussian noise.

    Parameters
    ----------
    duration : float, optional
        Duration of noise to be generated, in seconds. It may change after
        genetarting the noise, depending on its sample frequency.

    noise : array-like, optional
        Alternative noise array already generated.

    psd : 2d array-like, (time_samples, psd_samples)
        Power Spectral Density of the non-white part of the noise.

    sf : int
        Sample frequency of the signal.

    random_seed : int or 1-d array_like, optional
        Seed for numpy.random.RandomState.

    Attributes
    ----------
    noise : array

    duration : float
        Duration of the noise in seconds.

    """
    def __init__(self, duration=None, noise=None, psd=None, sf=None, random_seed=None):
        self.duration = duration  # May be corrected after calling _gen_noise()
        self.noise = np.asarray(noise) if duration is None else None
        self.psd = psd
        self.sf = sf
        self.random_seed = random_seed

        self._check_initial_parameters()
        
        # Generate new noise.
        if duration is not None:
            self._gen_noise()
        # Imported an already generated array.
        else:
            self.duration = len(noise) / sf

    def __getitem__(self, key):
        """Direct slice access to noise array."""
        return self.noise[key]

    def __len__(self):
        """Length of the noise array."""
        return len(self.noise)

    def __repr__(self):
        args = (type(self).__name__, self.duration, self.sf, self.random_seed)

        return "{}(t={}, sf={}, random_seed={})".format(*args)

    def add_to(self, x, snr=1, snr_lim=None, pos=0, sf=cfg.SF, l2_normed=True):
        """Add the simulated noise to the signal 'x'.

        Parameters
        ----------
        x : array
            Signal array. Its length must be lower or equal to
            the length of the noise array.

        snr : int or float, optional
            Signal to Noise Ratio. Defaults to 1.

        snr_lim : tuple, optional
            Limits in 'x' where to calculate the SNR.

        pos : int, optional
            Index position in the noise array where to inject the signal.
            0 by default.

        sf : int, optional
            Sample frequency of the signal.

        l2_normed : boolean, optional
            Normalize 'x' after adding the noise.
            True by default.

        Returns
        -------
        noisy : array
            Signal array with noise at the desired SNR.

        scale : float
            Coefficient used to rescale the signal.

        """
        n = len(x)
        if n > len(self.noise):
            raise ValueError("'x' is larger than the noise array")

        if snr_lim is None:
            scale = snr / self.snr(x, at=1/sf)
        else:
            scale = snr / self.snr(x[slice(*snr_lim)], at=1/sf)

        x_noisy = x * scale + self.noise[pos:pos+n]

        if l2_normed:
            norm = np.linalg.norm(x_noisy)
            x_noisy /= norm
            scale /= norm

        return (x_noisy, scale)

    def rescale(self, x, snr=1, sf=cfg.SF):
        """Rescale the signal 'x' to the given snr w.r.t. the PSD.

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

    def _check_initial_parameters(self):
        # Check optional arguments.
        if self.duration is not None:
            if not isinstance(self.duration, (int, float)):
                raise TypeError("'duration' must be an integer or float number")
        elif self.noise is None:
            raise TypeError("either 'duration' or 'noise' must be provided!")
        elif not isinstance(self.noise, (list, tuple, np.ndarray)):
            raise TypeError("'noise' must be an array-like iterable")

        # Check required arguments.
        if self.psd is None:
            raise TypeError("'psd' must be provided")
        if self.sf is None:
            raise TypeError("'sf' must be provided")

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