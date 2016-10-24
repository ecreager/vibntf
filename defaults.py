import numpy

__author__ = 'ecreager'

# stft
N = 1024
HOP = N//4

# ddm
Q = 2
L = 5
R = 50

# ntf
S = 2
Z = 3
EM_ITER = 100

# synth
FS = 44100 # music
SIG_DUR = 2.0 # seconds
F0 = 220.
RATE = 10.
DEPTH = F0/20.
PHASE_INIT = -numpy.pi/2  # phase of a cosine; theta = -pi/2 -> f(theta) = 0 and rising
FM_PHASE_INIT = -numpy.pi/2
N_PARTIALS = 5  # default argument to fm_square_wave (randomized during experiments)
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
DO_NOISE = True  # adding a bit of noise to the mixture signal helps with DDM stability
NOISING_SNR = 60
SAVE_AUDIO = True  # write audio ins/outs to disk
SCREEN = True  # apply heuristic threshold to noisy DDM estimates
P_FT_POWERED_PCNT = 10  # heuristic thresholding: lower bound
ABS_RATIOS_PCNT = 100  # heuristic thresholding: upper bound
SAVE_OBS = True  # save pobs(f,t,r) and q_init to disk
SAVE_COMPARE = True  # save comparison between, e.g., nmf and vibntf
ALGS = ['nmf', 'vibntf']  # default comparison in a synthetic data experiment
N_SEPARATIONS = 1
VIBNTF_DIR = '/tmp/vibntf'  # where to store data

EVAL_NMF_INIT = True  # for VibNTF with NMF initialization, also evaluate the initialization as a separator