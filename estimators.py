import numpy as np


#-----------------------------------------------------------------------
# Similarity test functions
#-----------------------------------------------------------------------

def mse(x, y):
    """Mean Squared Error."""
    return np.mean((x-y)**2)


def ssim(x, y):
    """Structural similarity index."""
    mux = x.mean()
    muy = y.mean()
    sx2 = x.var()
    sy2 = y.var()
    sxy = np.cov(x, y, ddof=0)[0, 1]
    l_ = 1
    c1 = (0.01*l_) ** 2
    c2 = (0.03*l_) ** 2

    return ((2*mux*muy+c1) * (2*sxy+c2)
            / ((mux**2+muy**2+c1) * (sx2+sy2+c2)))


def dssim(x, y):
    """Structural dissimilarity."""
    return (1 - ssim(x, y)) / 2


def residual(x, y):
    """Norm of the difference between 'x' and 'y'."""
    return np.linalg.norm(x - y)


def softmax(x, axis=None):
    """Softmax probability distribution."""
    coefs = np.exp(x)
    return coefs / coefs.sum(axis=axis, keepdims=True)


#-----------------------------------------------------------------------
# Classification
#-----------------------------------------------------------------------

def classificate(parents, children):
    """Classification algorithm.

    Method used to classificate a waveform by comparing the similarity of
    reconstructions made with different dictionaries.

    PARAMETERS
    ----------
    parents: array_like, (waveforms, features)
        Parent waveforms whose indices coincide to their respective morphological
        families. Each parent waveform will have associated 'len(parents)' children.

    children: array_like, (parents, waveforms, features)
        Reconstructions associated with parent waveforms.

    RETURNS
    -------
    index: int
        Index of the most fitting morphological family (same index as the parent
        waveform).

    """
    prods = []
    for ip, p in enumerate(parents):
        losses = []
        for c in children[:,ip]:
            if not np.isnan(np.sum(c)):
                losses.append(dssim(p, c))
            else:
                losses.append(1.0)  # Worst result
        prods.append(np.prod(losses))
    index = np.argmin(prods)
    
    return index