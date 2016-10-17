import numpy
import defaults
import util

__author__ = 'ecreager'


def hannd(M):
    """first derivative of length-M hann window"""
    wd = (numpy.pi/(M-1))*numpy.sin(2*numpy.pi*numpy.arange(M)/(M-1))
    return wd


def stddm(x, Q=None, L=None, N=None, M=None, fs=None, hop=None, do_print=defaults.DO_PRINT):
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
    wd = hannd(M)
    X, x_bufs = util.ec_stft(x, N=N, hop=hop, M=M)
    F, T = X.shape
    pad1 = (M + 1) / 2
    pad2 = M / 2
    xpad = numpy.hstack((numpy.zeros(pad1, ), x, numpy.zeros(pad2, )))
    wpad = numpy.zeros_like(xpad)
    inds = pad1 + hop * (1 + numpy.arange(T))
    tau = numpy.arange(N, dtype='float64') / fs  # time
    k = numpy.arange(Q) + 1
    k = k[numpy.newaxis, :]
    Aft = numpy.zeros((F, T, L, 2), dtype=numpy.complex_)
    bft = numpy.zeros((F, T, L, 1), dtype=numpy.complex_)
    x_bufs = x_bufs.T
    sigMats = x_bufs[:, :, None] * (k * (tau[:, None] ** (k - 1)))[None, :, :]
    sigPolyWinFfts = numpy.fft.fft((w[None, :, None] * sigMats), n=N, axis=1)
    binFreqs = 2 * numpy.pi * numpy.arange(N) / (N - 1)  # in rad
    binFreqsHz = fs * numpy.arange(N, dtype=numpy.float_) / (N - 1)  # in Hz
    sigWinDInnerProds = -(numpy.fft.fft((x_bufs * wd[None, :]), n=N, axis=1) - 1j *
                          fs * binFreqs[None, :] *
                          numpy.fft.fft((x_bufs * w[None, :]), n=N, axis=1))[:, :, None]
    for t, ind in enumerate(inds):
        for f in xrange(F):
            if False:
                bottomBin = int(numpy.floor(f - (L - 1) / 2.))
                topBin = int(numpy.floor(f + (L - 1) / 2.))
                nhdBins = numpy.arange(bottomBin, topBin + 1)
            Aft[f, t, :, :] = sigPolyWinFfts[t, xrange(f - 1 - (L - 1) / 2, f + (L - 1) / 2), :]
            bft[f, t, :, :] = sigWinDInnerProds[t, xrange(f - 1 - (L - 1) / 2, f + (L - 1) / 2), :]
    AHft = numpy.transpose(Aft, axes=(0, 1, 3, 2)).conj()
    AHAinv = fast2x2inv(numpy.sum(Aft[:, :, None, :, :] * AHft[:, :, :, :, None], axis=3))
    AHb = numpy.sum(AHft[:, :, :, :, None] * bft[:, :, None, :, :], axis=3)
    eta = numpy.sum(AHAinv[:, :, :, :, None] * AHb[:, :, None, :, :], axis=3).squeeze()
    st_freqs = numpy.imag(eta[:, :, 0] / (2 * numpy.pi))
    st_freqs_der = numpy.imag(eta[:, :, 1] / numpy.pi)
    return st_freqs, st_freqs_der, X, x_bufs


def fast2x2inv(A):  # quickly invert the 2x2 matrix A
    Z = numpy.asarray([[A[:, :, 1, 1], -A[:, :, 0, 1]], [-A[:, :, 1, 0], A[:, :, 0, 0]]]) / (
        A[:, :, 1, 1] * A[:, :, 0, 0] - A[:, :, 0, 1] * A[:, :, 1, 0])
    return numpy.transpose(Z, axes=(2, 3, 0, 1))


def unfold(a):
    if a.ndim > 1:
        z = numpy.zeros((2 * a.shape[0], a.shape[1]))
        z[0::2, :] = a.real
        z[1::2, :] = a.imag
    else:
        z = numpy.zeros(2 * len(a), )
        z[0::2] = a.real
        z[1::2] = a.imag
    return z


def fold(z):
    if z.ndim > 1:
        a = z[0::2, :] + 1j * z[1::2, :]
    else:
        a = z[0::2] + 1j * z[1::2]
    return a


def cmat(z):  # complex to real 2x2
    return z.real * numpy.eye(2) + numpy.flipud(numpy.diag([1, -1])) * z.imag


def Cmat(Z):  # complex NxM to real 2Nx2M
    return numpy.vstack([numpy.hstack([cmat(x) for x in z]) for z in Z])


def get_Lam(lam):  # unfolded penalty matrix
    d = numpy.zeros(2 * len(lam), )
    d[0::2] = lam
    d[1::2] = lam
    return numpy.diag(d)


def get_U(Q):  # imag projection matrix
    d = numpy.zeros(2 * Q, )
    d[1::2] = 1
    return numpy.diag(d)


def get_kappa(eta):  # constraint on freq and freq der
    return numpy.array([0, numpy.around(eta[0].imag, 1), 0, 0])


def get_kappa_bins(bin_freqs):
    return [numpy.array([0, 2 * numpy.pi * b, 0, 0], ndmin=2).T for b in bin_freqs]