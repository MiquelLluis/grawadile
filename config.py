#!/usr/bin/env python
#
# config.py
#
# Configuration file and default values (deprecated, will be removed).
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

# ---- Glitch samples generation ----
TH = 0.01   # Signal Threshold (amplitude min/max)
