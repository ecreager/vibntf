import copy
import numpy
import ddm
import defaults
from scipy.io import wavfile
import scipy.signal, scipy.io.wavfile

__author__ = 'ecreager'

ALL_NOTES = ['A0','A#0','B0',\
    'C1','C#1','D1','D#1','E1','F1','F#1','G1','G#1','A1','A#1','B1',\
    'C2','C#2','D2','D#2','E2','F2','F#2','G2','G#2','A2','A#2','B2',\
    'C3','C#3','D3','D#3','E3','F3','F#3','G3','G#3','A3','A#3','B3',\
    'C4','C#4','D4','D#4','E4','F4','F#4','G4','G#4','A4','A#4','B4',\
    'C5','C#5','D5','D#5','E5','F5','F#5','G5','G#5','A5','A#5','B5',\
    'C6','C#6','D6','D#6','E6','F6','F#6','G6','G#6','A6','A#6','B6',\
    'C7','C#7','D7','D#7','E7','F7','F#7','G7','G#7','A7','A#7','B7',\
    'C8','C#8','D8','D#8','E8','F8','F#8','G8','G#8','A8','A#8','B8']


def observe(x=None, N=None, hop=None, M=None, Q=None, L=None, R=None, fs=None, do_print=defaults.DO_PRINT):
    if N is None:
        if M is None:
            N = defaults.N
            M = N
        else:
            N = M
    else:
        if M is None:
            M = N
    if hop is None:
        hop = M/4
    w = numpy.hanning(M)
    wd = ddm.hannd(M)
    if Q is None:
        Q = defaults.Q
    if L is None:
        L = defaults.L
    if R is None:
        R = defaults.R
    R = R + (R - 1) % 2  # make odd so one bin can be exactly ratio = 0
    p = dict()
    p['R'] = R  # in case R becomes odd
    freqs, freqs_der, X, x_bufs = ddm.stddm(x, Q=Q, L=L, N=N, M=M, fs=fs, hop=hop, do_print=do_print)
    p['ft'] = normalize(numpy.abs(X))
    p['good_rats'] = get_good_rats(p_ft=p['ft'], L=L, fs=fs, N=N, freqs=freqs, freqs_der=freqs_der)
    ratios = freqs_der/freqs
    p['ratios'] = ratios  # unthresholded, undicretized ratios
    p['freqs'] = freqs  # for debugging; remove later
    p['freqs_der'] = freqs_der # for debugging; remove later
    good_ratios = ratios*p['good_rats']
    # quantize
    min_max_ratio = min((numpy.max(good_ratios), numpy.max(-good_ratios)))
    ratios[numpy.abs(ratios) > min_max_ratio] = min_max_ratio # clip
    ratio_bins = numpy.linspace(-min_max_ratio, min_max_ratio, num=R)
    if defaults.SCREEN:
        ratios = ratios*p['good_rats']
    p['r'] = numpy.reshape(numpy.digitize(ratios.flatten(), ratio_bins) -1, p['ft'].shape)
    p['ftr'] = numpy.zeros(p['r'].shape + (R, ))
    for f in xrange(p['r'].shape[0]):
        for t in xrange(p['r'].shape[1]):
            p['ftr'][f, t, p['r'][f, t]] = p['ft'][f, t]
    return p


def seps_observe(seps, N=None, hop=None, M=None):
    """p_ft of pre-mixed sources for supervised nmf"""
    return [normalize(numpy.abs(ec_stft(seps[:, i], N=N, hop=hop, M=M)[0])) for i in xrange(seps.shape[1])]


def load_sound(sound_fn=None, fs_expected=defaults.FS):
    fs, data = scipy.io.wavfile.read(sound_fn)
    if numpy.issubdtype(data.dtype, float):
        max_val = numpy.finfo(data.dtype).max  # don't bother changing bit depth for float .wavs
    elif numpy.issubdtype(data.dtype, int):
        max_val = numpy.iinfo(data.dtype).max
        data = data.astype(numpy.float_) / max_val
    else:
        raise Exception('non-int, non-float .wav file')
    # todo proper resampling if fs != fs_expected
    return fs, data


def save_sound(sound_fn=None, sound=None, fs=None):
    if fs is None:
        fs = defaults.FS
    wavfile.write(filename=sound_fn, rate=fs, data=sound)
    return


def ec_stft(x=None, N=None, hop=None, M=None):
    if N is None:
        if M is None:
            N = defaults.N
            M = N
        else:
            N = M
    else:
        if M is None:
            M = N
    if hop is None:
        hop = N/4
    window = numpy.hanning(M)
    x = numpy.squeeze(x)
    window = numpy.squeeze(window)
    M = window.shape[0]
    hM1 = (M+1)/2
    hM2 = M/2
    pad1 = M - hop
    pad2 = M - hop
    xpad = numpy.hstack((numpy.zeros(pad1, ), x, numpy.zeros(pad2, )))
    wpad = numpy.zeros_like(xpad)
    nHops = int(numpy.floor((len(xpad) - M - 0.)/hop))
    X = numpy.zeros((N/2+1, nHops), dtype=numpy.complex_)
    x_bufs = numpy.zeros((M, nHops))
    hs = numpy.arange(1, nHops+1)
    inds = hM1 + hop*hs
    for h, ind in zip(hs, inds):
        wpad[ind-hM1:ind+hM2] += window
        X[:, h-1] = numpy.fft.fft(window * xpad[ind-hM1:ind+hM2], n=N)[:N/2+1]
        x_bufs[:, h-1] = xpad[ind-hM1:ind+hM2]
    else:
        wolmax = numpy.max(wpad)*(N/2+1)/N
    X = X/wolmax
    return X, x_bufs


def ec_istft(X=None, M=None, hop=None):
    nBins, nHops = X.shape
    if M is None:
        M = 2*(nBins-1)
    hM1 = (M+1)/2
    hM2 = M/2
    pad1 = M - hop
    pad2 = M - hop
    xpad = numpy.hstack((numpy.zeros(pad1, ), numpy.zeros(nHops*hop, ), numpy.zeros(pad2, )))
    ind = hM1  # middle of first window
    buf = numpy.real(numpy.fft.ifft(numpy.hstack((X[:, 0], numpy.conj(X[-2:0:-1, 0])))))[:M]
    xpad[ind-hM1:ind+hM2] = buf  # first frame
    hs = numpy.arange(1, nHops+1)
    inds = hM1 + hop*hs
    for h, ind in zip(hs, inds):
        xpad[ind-hM1:ind+hM2] += numpy.real(numpy.fft.ifft(\
                numpy.hstack((X[:, h-1], numpy.conj(X[-2:0:-1, h-1])))\
            ))[:M]
    x = xpad[pad1:ind-hM1+hop]
    return x


def normalize(x, axis=None):
    return x / (numpy.sum(x, axis, keepdims=True) + numpy.spacing(1))


def normalize_and_output(x, out, axis=None):
    numpy.divide(x, numpy.sum(x, axis, keepdims=True) + numpy.spacing(1), out=out)


def get_good_rats(p_ft=None, L=None, fs=None, N=None, freqs=None, freqs_der=None):
    if defaults.SCREEN: # actually do something
        screen = numpy.zeros_like(p_ft, dtype=numpy.float_)
        screen[L+2:-L-2, 2:-2] = 1. # ddm still has boundary issues
        F, T = p_ft.shape
        p_ft_powered = p_ft**0.4
        thresh = numpy.percentile(p_ft_powered, defaults.P_FT_POWERED_PCNT)
        freq_deviation_thresh = 4*fs
        screen[numpy.abs(freqs - numpy.tile(fs/float(N)*numpy.arange(N/2+1)[:, numpy.newaxis], (1, T))) > freq_deviation_thresh] = 0 # keep estimated freqs in sensible range
        abs_ratios = numpy.abs(freqs_der/freqs)
        abs_ratios *= screen
        max_ratios_thresh = numpy.percentile(abs_ratios, defaults.ABS_RATIOS_PCNT)
        # apply various thresholds
        screen[p_ft_powered < thresh] = 0.
        screen[abs_ratios > max_ratios_thresh] = 0
        return screen
    else:
        return numpy.ones_like(p_ft, dtype=numpy.float_)


def apply_screen(X=None, screen=None):
    Xc = copy.deepcopy(X)
    if X.dtype == numpy.int_:
        Xc = numpy.asarray(Xc, dtype='float')
    Xc[screen == 0] = float('nan')
    return Xc


def center_and_segment(s, desired_length_samps):
    s_center = center_of_energy(s)
    half_desired_length_samps = desired_length_samps/2
    s_left_half_cropped = s[max(0, s_center-half_desired_length_samps):s_center]
    s_right_half_cropped = s[s_center:s_center+half_desired_length_samps]
    left_pad = numpy.zeros((max(0, half_desired_length_samps - len(s_left_half_cropped))), )
    right_pad = numpy.zeros((max(0, half_desired_length_samps - len(s_right_half_cropped))), )
    s = numpy.hstack((left_pad, s_left_half_cropped, s_right_half_cropped, right_pad))
    return s


def mix_files(sound_filename_1, sound_filename_2, snr=0, fs=defaults.FS, pad=True, do_ramp=True, desired_length_sec=defaults.SIG_DUR):
    _, s1 = load_sound(sound_filename_1, fs_expected=fs)
    _, s2 = load_sound(sound_filename_2, fs_expected=fs)
    return mix(s1, s2, snr=0, fs=defaults.FS, do_ramp=True, desired_length_sec=desired_length_sec)


def mix(s1, s2, snr=0, fs=defaults.FS, do_ramp=True, desired_length_sec=defaults.SIG_DUR):
    if s1.ndim > 1:
        s1 = s1[:, 0]
    if s2.ndim > 1:
        s2 = s2[:, 0]
    desired_length_samps = int(fs*desired_length_sec)
    s1 = center_and_segment(s1, desired_length_samps)
    s2 = center_and_segment(s2, desired_length_samps)
    s1 = s1/numpy.std(s1)
    s2 = s2/numpy.std(s2)
    if do_ramp: # 10 msec ramp
        ramp_length_msec = 10.
        ramp_length_samps = int(fs*ramp_length_msec/1000)
        s1 = ramp(s1, ramp_length_samps)
        s2 = ramp(s2, ramp_length_samps)
    s1 *= 10**(-float(snr)/20)
    seps = numpy.vstack((s1, s2)).T
    mix = numpy.sum(seps, axis=1)
    max_val = numpy.max(numpy.abs(mix))
    seps = seps/max_val
    mix = mix/max_val
    return mix, seps


def ramp(x, ramp_length_samps):
    ramp = numpy.ones_like(x)
    ramp[:ramp_length_samps] = numpy.linspace(0, 1, ramp_length_samps)
    ramp[-ramp_length_samps:] = 1.-numpy.linspace(0, 1, ramp_length_samps)
    return x*ramp


def get_freq_val(note):
    note = note.upper()
    try:
        note_idx = ALL_NOTES.index(note)
    except Exception:
        raise Exception
    A4_idx = ALL_NOTES.index('A4')
    n_half_steps = note_idx - A4_idx
    return 440*2**(n_half_steps/12.)


def get_note_name(freq):
    n_half_steps = 12*numpy.log2(freq/440.) # halfsteps away from A4
    A4_idx = ALL_NOTES.index('A4')
    note_idx = A4_idx + int(round(n_half_steps))
    try:
        return ALL_NOTES[note_idx]
    except Exception:
        raise Exception


def KL(A, B):
    Atemp = A / sum(A)
    Btemp = B / sum(B)
    return sum(Atemp * numpy.log(Atemp / Btemp + numpy.spacing(1)))


def center_of_energy(x):
    """sample index for energy mean"""
    return int(numpy.sum(numpy.arange(len(x))*normalize(x**2)))