import copy
import os
import numpy
import sys
from util import normalize, ec_istft, save_sound

__author__ = 'ecreager'


def nmf(p, Z, em_iterations=100, q_fixed=dict(), q_init=None):
    F, T = p['ft'].shape
    if q_init is None:
        q = init_nmf(F, T, Z)
    else:
        q = copy.deepcopy(q_init)
    for i in range(em_iterations):
        q['ft'] = q['f|z'].dot(q['zt'])
        rho = p['ft']/(q['ft']+numpy.spacing(1))
        f_scale_factor = rho.dot(q['zt'].T)
        q['f|z'] = normalize(f_scale_factor*q['f|z'], axis=0)
        q['ft'] = q['f|z'].dot(q['zt'])
        t_scale_factor = q['f|z'].T.dot(rho)
        q['zt'] *= t_scale_factor
        q.update(q_fixed)
    q['ft'] = q['f|z'].dot(q['zt'])
    return q


def init_nmf(F, T, Z, q_init=None):
    if q_init is None:
        q = dict()
        q['f|z'] = normalize(1+numpy.random.exponential(size=(F, Z)), axis=0)
        q['zt'] = normalize(1+numpy.random.exponential(size=(Z, T)))
    else:
        q = copy.deepcopy(q_init)
    return q


def init_ntf(F, T, R, Z, S, q_init=None):
    if q_init is None:
        q = dict()
        q['s'] = normalize(1+numpy.random.exponential(size=(S, )))
        q['f|sz'] = normalize(1+numpy.random.exponential(size=(F, S, Z)), axis=0)
        q['zt|s'] = normalize(1+numpy.random.exponential(size=(Z, T, S)), axis=(0, 1))
        q['r|ts'] = normalize(1+numpy.random.exponential(size=(R, T, S)), axis=0)
        q['ft|s'] = numpy.zeros((F, T, S)) # redundant but keep in for debugging
        q['ftr'] = numpy.zeros((F, T, R))
    else:
        q = copy.deepcopy(q_init)
    return q


def fm_ntf(p, S=None, Z=None, em_iterations=10, q_fixed=dict(), good_rats=None, q_init=None):
    F, T, R = p['ftr'].shape
    if q_init is None:
        q = init_ntf(F, T, R, Z, S, q_init)
    else:
        q = copy.deepcopy(q_init)
    if good_rats is None:
        q['good_rats'] = numpy.ones((F, T))
    else:
        q['good_rats'] = good_rats
    for s in range(S):
        q['ft|s'][:, :, s] = numpy.dot(q['f|sz'][:, s, :], q['zt|s'][:, :, s]) # redundant
        q['ftr'] += q['s'][s]*q['r|ts'][:, :, s].T[numpy.newaxis, :, :]*q['ft|s'][:, :, s, numpy.newaxis]
    for i in range(em_iterations):  # E-M
        sys.stdout.write('.')
        sys.stdout.flush()
        F, T, R = p['ftr'].shape
        gcm_likelihood = numpy.zeros(
            (F, T, S))  # likelihood contribution from the grouping cues model (i.e., pitch modulation modeling)
        sm_likelihood = numpy.zeros((F, T, S))  # likelihood contribution from the source model (i.e., NMF)
        inds = numpy.indices((F, T))
        # E
        for s in range(S):
            gcm_likelihood[inds[0].flatten(), inds[1].flatten(), s] = q['r|ts'][p['r'].flatten(), inds[1].flatten(), s]
            q['ft|s'][:, :, s] = numpy.dot(q['f|sz'][:, s, :], q['zt|s'][:, :, s])
            sm_likelihood[:, :, s] = q['s'][s] * q['ft|s'][:, :, s]
        qfts = gcm_likelihood * sm_likelihood
        rho = (qfts * p['ft'][:, :, None]) / (
            numpy.sum(qfts, axis=2, keepdims=True) + numpy.spacing(1))  # rho = p^obs(f, t)/q(s|f, t)
        # M
        for s in range(S):
            common_factor = rho[:, :, s] / (q['ft|s'][:, :, s] + numpy.spacing(1))
            # update (f|s, z)
            f_scale_factor = common_factor.dot(q['zt|s'][:, :, s].T)
            q['f|sz'][:, s, :] = normalize(q['f|sz'][:, s, :] * f_scale_factor, axis=0)
            # update q(z, t|s)
            zt_scale_factor = q['f|sz'][:, s, :].T.dot(common_factor)
            q['zt|s'][:, :, s] = normalize(q['zt|s'][:, :, s] * zt_scale_factor, axis=(0, 1))
            # update q(r|t, s)
            for r in range(R):
                q['r|ts'][r, :, s] = numpy.sum(rho[:, :, s] * (p['r'] == r), axis=0)
            q['r|ts'] = normalize(q['r|ts'], axis=0)
            # update q(s)
            q['s'] = normalize(numpy.sum(rho, axis=(0, 1)))
    return q


def cross_entropy(pobsft, rft, q):
    assert numpy.all(q >= 0), 'approx distribution not nonnegative!'
    F, T, R = q.shape
    q[q <= 0] = numpy.spacing(1) # avoids nan cross_entropy but maybe breaks monotonic increase of objective
    ce = 0.
    for r in range(R):
        if numpy.count_nonzero(r==rft) > 0:
            qft = q[:, :, r]
            ce += numpy.sum(pobsft[rft==r].flatten()*numpy.log(qft[rft==r].flatten()))
    return ce


def get_mask(q):
    q_fts = q['ft|s']*q['s'][numpy.newaxis, numpy.newaxis, :]
    return q_fts/(numpy.sum(q_fts, axis=2)[:, :, numpy.newaxis] + numpy.spacing(1))  # q_sIft


def get_nmf_mask(q):
    q_fts = numpy.dstack([q['f|z'][:, i, None]*q['zt'][i, None, :] for i in range(q['zt'].shape[0])])
    return q_fts/(numpy.sum(q_fts, axis=2)[:, :, numpy.newaxis] + numpy.spacing(1))  # q_sIft


def apply_mask(X, mask, save_dir=None, do_save=False, M=None, hop=None, fs=None):
    y = []
    F, _, S = mask.shape
    N = 2*(F-1)
    if M is None:
        M = N
    if hop is None:
        hop = N/4
    X_masked = X[:, :, numpy.newaxis]*mask
    y = numpy.concatenate(
            [ec_istft(X=X_masked[:, :, s], M=M, hop=hop)[:, numpy.newaxis]
             for s in range(S)], axis=1)
    if do_save:
        print('\nsaving separated outputs in %s' % save_dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for s in range(S):
            fn = 'sep-' + str(s) + '.wav'
            save_sound(sound=y[:, s], sound_fn=save_dir+'/'+fn, fs=int(fs))
    return y, X_masked
