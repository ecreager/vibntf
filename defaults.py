import numpy

__author__ = 'ecreager'

# stft
N = 1024
HOP = N/4

# ddm
Q = 2
L = 5
R = 50

# ntf
S = 2
Z = 3
# EM_ITER = 100
EM_ITER = 5

# synth
FS = 44100. # music
SIG_DUR = 2.0 # seconds
F0 = 220.
RATE = 10.
DEPTH = F0/20.
PHASE_INIT = 0
FM_PHASE_INIT = -numpy.pi/2
N_PARTIALS = 5  # default argument to fm_square_wave (randomized during experiment)
SIG_LEN = int(SIG_DUR*FS)
SOME_NOTES = True
# value range of random params for vibrato square wave
SOME_NOTES_RANGE = [36, 72]  # A3--G#5
DEPTH_TO_FUND_RANGE = [1/20., 1/5.]  # fraction of fundamental
N_PARTIALS_RANGE = [10, 30]
RATE_RANGE = [0.5, 10.]  # Hz

# experiments
MIX_DUR = SIG_DUR  # seconds
MIX_SNR = 0.
DO_NOISE = True
NOISING_SNR = 60  # add noise to the observations
STORE_GAINS = False
DO_PRINT = True # print info, e.g., ddm x/X
DO_WRITE = True
SCREEN = True  # apply heuristic threshold to noisy DDM estimates
P_FT_POWERED_PCNT = 10  # lower bound
ABS_RATIOS_PCNT = 100  # upper bound
SAVE_OBS = True
EVAL_NMF_INIT = True  # for VibNTF with NMF initialization, also evaluate the initialization as a separator
SAVE_COMPARE = True

# experiments
ALGS = ['nmf', 'vibntf']  # default comparison in a synthetic data experiment
N_SEPARATIONS = 1
VIBNTF_DIR = '/tmp/vibntf'  # where to store data

STORE_GAINS = False
SAVE_FACTORS = False
NEW_UPDATES = True