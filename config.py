#!/usr/bin/env python
#
# config.py
#
# Configuration file and default values.
#


# ---- Global settings ----
SF = 16384  # Sample frequency (Hz)
ST = 1/SF   # Sample time step (s)


# ---- Limits for the wave functions ----
# Set limits. LIM0 corresponds to the first set from where all dictionaries are
# trained.
#               min          Max
LIM0   = dict(mf0=40,      Mf0=1500,    # Frequency (Hz)
              mhrss=5e-22, Mhrss=4e-21, # Root sum squared (Hz^-1/2)
              mQ=2,        MQ=20,       # Quality factor
              mT=0.001,    MT=0.01)     # Duration (G)
# LIM1SG = dict(mf0=380,     Mf0=420,
#               mhrss=1e-21, Mhrss=5e-21,
#               mQ=5,        MQ=10,
#               mSNR=1,      MSNR=400)    # Signal to Noise Ratio
# LIM1G  = dict(mhrss=1e-21, Mhrss=5e-21, 
#               mT=0.001,    MT=0.01,
#               mSNR=1,      MSNR=400)
# LIM3   = dict(mf0=40,      Mf0=1500, 
#               mhrss=5e-22, Mhrss=4e-21,
#               mQ=2,        MQ=20,
#               mT=0.001,    MT=0.01)


# ---- Glitch samples generation ----
TH = 0.01   # Signal Threshold (amplitude min/max)


# ---- Dictionary learning step ----
# Number of glitches from the main set to be used for training each wf dict
TRAIN_N = 360
# Number of dictionary atoms for each waveform (SG, G, RD)
DICT_N = (250, 250, 250)
# Quantity FRACTION: patch_number / number_of_atoms
PATCH_N = 100
# Minimum points of the wave to include in each patch
PATCH_MIN = 16  # pot. de 2 mínima per representar 1 periode del senyal de més freqüència
# Number of iterations to solve the dictionary learning problem
DICT_ITERS = 1000
# Alpha (lambda) parameter
DICT_ALPHA = .2


# ---- Parameter testing ----
# Number of samples from the main set to be used for testing
TEST_N = 40
# Transform method
DICT_METHOD = 'lasso_lars'
# Window step for signal reconstruction.
STEP = 2
# NEW DIFFERENT ALPHA/LAMBAD PARAMETER FOR EACH WF KIND
DICT_ALPHAS = (.3, .07, .1)


# WF names
WF = ('SG', 'G', 'RD')