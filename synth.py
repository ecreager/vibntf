import numpy
import util
import defaults

__author__ = 'ecreager'


def fm_square_wave(f0=defaults.F0, depth=defaults.DEPTH, rate=defaults.RATE, n_partials=defaults.N_PARTIALS, sig_len=defaults.SIG_LEN, fs=defaults.FS, phase_init=defaults.PHASE_INIT, fm_phase_init=defaults.FM_PHASE_INIT):
    sig_len = int(sig_len)
    t = numpy.arange(sig_len, )/fs
    harm_inds = numpy.arange(start=1, stop=n_partials*2, step=2)
    max_freqs = harm_inds*f0 + harm_inds*depth
    harm_inds = harm_inds[max_freqs < fs/2]
    n_partials = len(harm_inds)
    ampls = 2/numpy.pi/harm_inds
    freq_laws = (f0 + depth/(2*numpy.pi)*numpy.cos(2*numpy.pi*rate*t[:, numpy.newaxis] + fm_phase_init))*harm_inds[numpy.newaxis, :]
    phases = numpy.zeros((sig_len, n_partials))
    phases[0, :] = fm_phase_init*harm_inds + phase_init
    for n in range(1, sig_len):
        phases[n, :] = phases[n-1, :] + freq_laws[n-1, :]*2*numpy.pi/fs
    x = numpy.dot(numpy.cos(phases), ampls)
    return x


def chirp(f_start=None, f_end=None, sig_len=None, fs=defaults.FS, phase_init=defaults.PHASE_INIT):   # linear chirp
    freq_law = numpy.linspace(f_start, f_end, num=sig_len)
    phase = numpy.zeros((sig_len, ))
    phase[0] = phase_init
    for n in range(1, int(sig_len)):
        phase[n] = phase[n-1] + freq_law[n-1]*2*numpy.pi/fs
    return numpy.cos(phase)


def rand_params(fs=defaults.FS, sig_dur=defaults.SIG_DUR):
    sig_len = int(fs*sig_dur)
    if defaults.SOME_NOTES:
        SOME_NOTES = util.ALL_NOTES[defaults.SOME_NOTES_RANGE[0]:defaults.SOME_NOTES_RANGE[1]]
        note_name = SOME_NOTES[numpy.random.randint(len(SOME_NOTES))]
    else:
        note_name = util.ALL_NOTES[numpy.random.randint(len(util.ALL_NOTES))]
    note_freq = util.get_freq_val(note_name)
    depth_range = [d*note_freq for d in defaults.DEPTH_TO_FUND_RANGE]
    n_partials_range = defaults.N_PARTIALS_RANGE
    rate_range = defaults.RATE_RANGE
    rate = 10**numpy.random.uniform(*numpy.log10(rate_range))
    depth = numpy.random.uniform(*depth_range)
    n_partials = numpy.random.randint(*n_partials_range)
    phase_init = numpy.random.uniform(0, 2*numpy.pi)
    vib_phase_init = numpy.random.uniform(0, 2*numpy.pi)
    return note_freq, depth, rate, n_partials, sig_len, fs, phase_init, vib_phase_init


def get_name_from_params(f0=None, depth=None, rate=None, n_partials=None, sig_len=None, fs=None, phase_init=None, fm_phase_init=None):
    return '%s-d%.f-r%.f' % (util.get_note_name(f0), depth, rate)
