import numpy as np
from numpy.random import uniform

from . import config as cfg


def gen_set(sq, sf=cfg.SF):
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
    mtSG = 2 * cfg.LIM0['MQ'] / (np.pi * cfg.LIM0['mf0']) * np.sqrt(-np.log(cfg.TH))
    mtG = cfg.LIM0['MT']
    mtRD = np.sqrt(2) * cfg.LIM0['MQ'] / (np.pi * cfg.LIM0['mf0']) * (-np.log(cfg.TH))
    # Maximum signal length (rounded UP)
    mslSG = int(np.ceil(mtSG * sf))
    mslG = int(np.ceil(mtG * sf))
    mslRD = int(np.ceil(mtRD * sf))

    # Waveform signals
    sigSG = np.zeros((sq[0], mslSG), dtype=float)  # Sinus Gaussian
    sigG = np.zeros((sq[1], mslG),  dtype=float)  # Gaussian
    sigRD = np.zeros((sq[2], mslRD), dtype=float)  # Ring-Down
    # Associated parameters
    sigSGp = np.empty((sq[0], 6), dtype=float)
    sigGp = np.empty((sq[1], 4), dtype=float)
    sigRDp = np.empty((sq[2], 6), dtype=float)
    # Sample time points (s) (not centered!)
    tp = np.arange(0, max(mtSG, mtG, mtRD), 1/sf)

    # Sample generation
    # ---- Sine Gaussian ----
    for i in range(sq[0]):
        f0, q, hrss, t = gen_params_0sg()

        sl = int(t*sf)  # Particular signal length
        signal = wave_sg(tp[:sl], t/2, f0, q, hrss)

        # Store the signal centered and normalized
        off = int((mtSG - t) / 2 * sf)  # Offset
        sigSG[i, off:off+sl] = signal / max(abs(signal))
        sigSGp[i] = (t, f0, q, hrss, off, off+sl)
    # ---- Gaussian ----
    for i in range(sq[1]):
        hrss, t = gen_params_0g()

        sl = int(t*sf)  # Particular signal length
        signal = wave_g(tp[:sl], t/2, hrss, t)

        # Store the signal centered and normalized
        off = int((mtG - t) / 2 * sf)  # Offset
        sigG[i, off:off+sl] = signal / max(abs(signal))
        sigGp[i] = (t, hrss, off, off+sl)
    # ---- Ring-Down ----
    for i in range(sq[2]):
        f0, q, hrss, t = gen_params_0rd()

        sl = int(t*sf)  # Particular signal length
        signal = wave_rd(tp[:sl], 0, f0, q, hrss)

        # Store the signal centered and normalized
        off = int((mtRD - t) / 2 * sf)  # Offset
        sigRD[i, off:off+sl] = signal / max(abs(signal))
        sigRDp[i] = (t, f0, q, hrss, off, off+sl)

    return (tp, (sigSG, sigG, sigRD), (sigSGp, sigGp, sigRDp))


# ---- Wave functions (ONLY X COMPONENT) ----
def wave_sg(t, t0, f0, Q, hrss):
    h0 = np.sqrt(np.sqrt(2) * np.pi * f0 / Q) * hrss
    env = h0 * np.exp(- (np.pi * f0 / Q * (t-t0)) ** 2)
    arg = 2 * np.pi * f0 * (t-t0)

    return np.sin(arg) * env


def wave_g(t, t0, hrss, T):
    h0 = (-8*np.log(cfg.TH))**(1/4) * hrss / np.sqrt(T)
    env = h0 * np.exp(4 * np.log(cfg.TH) * ((t-t0) / T)**2)

    return env


def wave_rd(t, t0, f0, Q, hrss):
    h0 = np.sqrt(np.sqrt(2) * np.pi * f0 / Q) * hrss
    env = h0 * np.exp(- np.pi / np.sqrt(2) * f0 / Q * (t - t0))
    arg = 2 * np.pi * f0 * (t - t0)

    return np.sin(arg) * env


# ---- Parameter generation ----
# All parameters for which Max/Min is greater than 1 order of magnitude (or its
# range is within two different orders, say [0.5, 1.8]) must be generated
# with a logarithmic random distribution.
#

def gen_params_0sg():
    f0 = int(10 ** uniform(np.log10(cfg.LIM0['mf0']), np.log10(cfg.LIM0['Mf0'])))  # Frequency
    Q = 10 ** uniform(np.log10(cfg.LIM0['mQ']), np.log10(cfg.LIM0['MQ']))  # Q factor
    hrss = 10 ** uniform(np.log10(cfg.LIM0['mhrss']), np.log10(cfg.LIM0['Mhrss']))  # hrss
    t = 2 * Q / (np.pi * f0) * np.sqrt(-np.log(cfg.TH))  # TOTAL duration (not centered)

    return (f0, Q, hrss, t)


def gen_params_0g():
    hrss = 10 ** uniform(np.log10(cfg.LIM0['mhrss']), np.log10(cfg.LIM0['Mhrss']))  # hrss
    t = uniform(cfg.LIM0['mT'], cfg.LIM0['MT'])  # TOTAL duration

    return (hrss, t)


def gen_params_0rd():
    f0 = int(10 ** uniform(np.log10(cfg.LIM0['mf0']), np.log10(cfg.LIM0['Mf0'])))  # Frequency
    Q = 10 ** uniform(np.log10(cfg.LIM0['mQ']), np.log10(cfg.LIM0['MQ']))  # Q factor
    hrss = 10 ** uniform(np.log10(cfg.LIM0['mhrss']), np.log10(cfg.LIM0['Mhrss']))  # hrss
    t = -np.sqrt(2) * Q / (np.pi * f0) * np.log(cfg.TH)  # Duration

    return (f0, Q, hrss, t)
