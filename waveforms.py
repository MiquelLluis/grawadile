import copy
import pickle

import numpy as np
from numpy import pi, exp, sqrt, log, log10, sin, ceil
from numpy.random import uniform

from . import config as cfg


# -----------------------------------------------------------------------------
#  MAIN CLASSES
# -----------------------------------------------------------------------------

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
        return sqrt(self.psd(f))

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

        return sqrt(4 * at**2 * af * sum_)

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
        nf_coefs = sqrt(length * sf * self.psd(f, margins=0)) / 2
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


class ConvergenceError(ValueError):
    pass


# ------------------------------------------------------------------------------
#  MAIN FUNCTIONS
# ------------------------------------------------------------------------------

def gen_set(sq):
    """Generates three sets of noise transients, one of each kind of waveform.

    Generates 'sq' noise transients for each kind of waveform: Sine Gaussisan,
    Gaussian, and Ring-Down. The signal length is the same between each
    waveform array, and it is computed with the extremest parameters that
    maximize the duration of the waveform.

    The parameters used for each signal are randomly generated within their
    respective limits, and stored in separated arrays.

    Parameters
    ----------
    sq : list of ints
        Amount of waveform signals to be generated for each kind of waveform.

    Returns
    -------
    tp : array
        Time points used for the evaluation of wave functions.
        tp[0] corresponds to signal[p0], not signal[0].

    (sampSG, sampG, sampRD) : array of shape (n_components, n_features)
        Sinus Gaussian, Gaussian and Ring-Down signals.

    (sampSGp, sampGp, sampRDp) : array
        Parameters associated to each signal, with shapes
            sampSGp[i] = [t, f0, q, hrss, p0, p1]
            sampGp[i]  = [t, hrss, p0, p1]
            sampRDp[i] = [t, f0, q, hrss, p0, p1]
        where 'p0' and 'p1' are the starting and ending indices of each waveform
        inside each signal.

    """
    # Maximum time length (maximum waveform length)
    mtSG = 2 * cfg.LIM0['MQ'] / (pi * cfg.LIM0['mf0']) * sqrt(-log(cfg.TH))
    mtG = cfg.LIM0['MT']
    mtRD = sqrt(2) * cfg.LIM0['MQ'] / (pi * cfg.LIM0['mf0']) * (-log(cfg.TH))
    # Maximum signal length (rounded UP)
    mslSG = int(ceil(mtSG * cfg.SF))
    mslG = int(ceil(mtG * cfg.SF))
    mslRD = int(ceil(mtRD * cfg.SF))

    # Waveform signals
    sigSG = np.zeros((sq[0], mslSG), dtype=float)  # Sinus Gaussian
    sigG = np.zeros((sq[1], mslG),  dtype=float)  # Gaussian
    sigRD = np.zeros((sq[2], mslRD), dtype=float)  # Ring-Down
    # Associated parameters
    sigSGp = np.empty((sq[0], 6), dtype=float)
    sigGp = np.empty((sq[1], 4), dtype=float)
    sigRDp = np.empty((sq[2], 6), dtype=float)
    # Sample time points (s) (not centered!)
    tp = np.arange(0, max(mtSG, mtG, mtRD), cfg.ST)

    # Sample generation
    # ---- Sine Gaussian ----
    for i in range(sq[0]):
        f0, q, hrss, t = gen_params_0sg()

        sl = int(t*cfg.SF)  # Particular signal length
        signal = wave_sg(tp[:sl], t/2, f0, q, hrss)

        # Store the signal centered and normalized
        off = int((mtSG - t) / 2 * cfg.SF)  # Offset
        sigSG[i, off:off+sl] = signal / max(abs(signal))
        sigSGp[i] = (t, f0, q, hrss, off, off+sl)
    # ---- Gaussian ----
    for i in range(sq[1]):
        hrss, t = gen_params_0g()

        sl = int(t*cfg.SF)  # Particular signal length
        signal = wave_g(tp[:sl], t/2, hrss, t)

        # Store the signal centered and normalized
        off = int((mtG - t) / 2 * cfg.SF)  # Offset
        sigG[i, off:off+sl] = signal / max(abs(signal))
        sigGp[i] = (t, hrss, off, off+sl)
    # ---- Ring-Down ----
    for i in range(sq[2]):
        f0, q, hrss, t = gen_params_0rd()

        sl = int(t*cfg.SF)  # Particular signal length
        signal = wave_rd(tp[:sl], 0, f0, q, hrss)

        # Store the signal centered and normalized
        off = int((mtRD - t) / 2 * cfg.SF)  # Offset
        sigRD[i, off:off+sl] = signal / max(abs(signal))
        sigRDp[i] = (t, f0, q, hrss, off, off+sl)

    return (tp, (sigSG, sigG, sigRD), (sigSGp, sigGp, sigRDp))


# ------------------------------------------------------------------------------
#  AUXILIAR FUNCTIONS
# ------------------------------------------------------------------------------

# ---- Wave functions (ONLY X COMPONENT) ----
def wave_sg(t, t0, f0, Q, hrss):
    h0 = sqrt(sqrt(2) * pi * f0 / Q) * hrss
    env = h0 * exp(- (pi * f0 / Q * (t-t0)) ** 2)
    arg = 2 * pi * f0 * (t-t0)

    return sin(arg) * env


def wave_g(t, t0, hrss, T):
    h0 = (-8*log(cfg.TH))**(1/4) * hrss / sqrt(T)
    env = h0 * exp(4 * log(cfg.TH) * ((t-t0) / T)**2)

    return env


def wave_rd(t, t0, f0, Q, hrss):
    h0 = sqrt(sqrt(2) * pi * f0 / Q) * hrss
    env = h0 * exp(- pi / sqrt(2) * f0 / Q * (t - t0))
    arg = 2 * pi * f0 * (t - t0)

    return sin(arg) * env


# ---- Parameter generation ----
# All parameters for which Max/Min is greater than 1 order of magnitude (or its
# range is within two different orders, say [0.5, 1.8]) must be generated
# with a logarithmic random distribution.
#

def gen_params_0sg():
    f0 = int(10 ** uniform(log10(cfg.LIM0['mf0']), log10(cfg.LIM0['Mf0'])))  # Frequency
    Q = 10 ** uniform(log10(cfg.LIM0['mQ']), log10(cfg.LIM0['MQ']))  # Q factor
    hrss = 10 ** uniform(log10(cfg.LIM0['mhrss']), log10(cfg.LIM0['Mhrss']))  # hrss
    t = 2 * Q / (pi * f0) * sqrt(-log(cfg.TH))  # TOTAL duration (not centered)

    return (f0, Q, hrss, t)


def gen_params_0g():
    hrss = 10 ** uniform(log10(cfg.LIM0['mhrss']), log10(cfg.LIM0['Mhrss']))  # hrss
    t = uniform(cfg.LIM0['mT'], cfg.LIM0['MT'])  # TOTAL duration

    return (hrss, t)


def gen_params_0rd():
    f0 = int(10 ** uniform(log10(cfg.LIM0['mf0']), log10(cfg.LIM0['Mf0'])))  # Frequency
    Q = 10 ** uniform(log10(cfg.LIM0['mQ']), log10(cfg.LIM0['MQ']))  # Q factor
    hrss = 10 ** uniform(log10(cfg.LIM0['mhrss']), log10(cfg.LIM0['Mhrss']))  # hrss
    t = -sqrt(2) * Q / (pi * f0) * log(cfg.TH)  # Duration

    return (f0, Q, hrss, t)
