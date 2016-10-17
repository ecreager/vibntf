import copy
import os
import numpy
import sys
import defaults  # todo: DELETE
from util import normalize, ec_istft, save_sound

__author__ = 'ecreager'


def nmf(p, Z, em_iterations=100, q_fixed=dict(), q_init=None):
    F, T = p['ft'].shape
    if q_init is None:
        q = init_nmf(F, T, Z)
    else:
        q = copy.deepcopy(q_init)
    for i in xrange(em_iterations):
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

def old_updates(q, p, gain, all_gains, S=None, Z=None, q_fixed=dict(), good_rats=None, store_gains=False):
    F, T, R = p['ftr'].shape
    if good_rats is None:
        q['good_rats'] = numpy.ones((F, T))
    else:
        q['good_rats'] = good_rats
    s_scale_factor = numpy.zeros_like(q['s'])
    f_scale_factor = numpy.zeros_like(q['f|sz'])
    r_scale_factor = numpy.zeros_like(q['r|ts'])
    zt_scale_factor = numpy.zeros_like(q['zt|s'])
    rho = numpy.zeros((F, T, R))
    # E
    q['ftr'][:] = 0
    for s in xrange(S):
        q['ft|s'][:, :, s] = numpy.dot(q['f|sz'][:, s, :], q['zt|s'][:, :, s])
        q['ftr'] += q['s'][s] * q['r|ts'][:, :, s].T[numpy.newaxis, :, :] * q['ft|s'][:, :, s,
                                                                            numpy.newaxis]  # without good_rats
    rho[:] = 0
    rho = p['ftr'] / (q['ftr'] + numpy.spacing(1))  # wasted divisions since most p['ftr'] are zero

    # M
    s_scale_factor[:] = 0
    f_scale_factor[:] = 0
    r_scale_factor[:] = 0
    zt_scale_factor[:] = 0
    # update f|zs
    B = numpy.transpose(q['r|ts'], axes=(1, 0, 2))[:, :, :, numpy.newaxis] * numpy.transpose(q['zt|s'], axes=(1, 2, 0))[
                                                                             :, numpy.newaxis, :, :] * q['s'][
                                                                                                       numpy.newaxis,
                                                                                                       numpy.newaxis, :,
                                                                                                       numpy.newaxis]
    f_scale_factor = numpy.tensordot(rho, B, axes=2)
    q['f|sz'] = normalize(q['f|sz'] * f_scale_factor, axis=0)
    if store_gains:
        all_gains.append(calc_gain(p, q))
    # update zt|s and mixing gains
    for s in xrange(S):
        phi = rho * q['r|ts'][:, :, s].T[numpy.newaxis, :, :]
        for r in xrange(R):
            zt_scale_factor[:, :, s] += q['s'][s] * numpy.dot(q['f|sz'][:, s, :].T, phi[:, :, r])
    q['zt|s'] = normalize(q['zt|s'] * zt_scale_factor, axis=(0, 1))
    if store_gains:
        all_gains.append(calc_gain(p, q))
    # recompute source models and update r|ts
    for s in xrange(S):
        q['ft|s'][:, :, s] = numpy.dot(q['f|sz'][:, s, :], q['zt|s'][:, :, s])
        r_scale_factor[:, :, s] = q['s'][s] * numpy.sum(rho * q['ft|s'][:, :, s, numpy.newaxis], axis=0).T
    q['r|ts'] = normalize(q['r|ts'] * r_scale_factor, axis=0)
    if store_gains:
        all_gains.append(calc_gain(p, q))
    if store_gains:
        all_gains.append(calc_gain(p, q))
    # update s
    for s in xrange(S):
        s_scale_factor[s] = numpy.sum(
            rho * q['r|ts'][:, :, s].T[numpy.newaxis, :, :] * q['ft|s'][:, :, s, numpy.newaxis])
    q['s'] = normalize(q['s'] * s_scale_factor)
    if store_gains:
        all_gains.append(calc_gain(p, q))
    q.update(q_fixed)
    if store_gains:
        gain.append(cross_entropy(p['ft'], p['r'], q['ftr']))

    return q, gain, all_gains

# @profile
def new_updates(u, p, ugain, uall_gains, S=None, Z=None, q_fixed=dict(), good_rats=None, store_gains=False):
    F, T, R = p['ftr'].shape
    gcm_likelihood = numpy.zeros((F, T, S))  # likelihood contribution from the grouping cues model (i.e., pitch modulation modeling)
    sm_likelihood = numpy.zeros((F, T, S))  # likelihood contribution from the source model (i.e., NMF)
    inds = numpy.indices((F, T))
    # E
    for s in xrange(S):
        gcm_likelihood[inds[0].flatten(), inds[1].flatten(), s] = u['r|ts'][p['r'].flatten(), inds[1].flatten(), s]
        u['ft|s'][:, :, s] = numpy.dot(u['f|sz'][:, s, :], u['zt|s'][:, :, s])
        if store_gains:
            u['ftr'] += u['s'][s] * u['r|ts'][:, :, s].T[numpy.newaxis, :, :] * u['ft|s'][:, :, s,
                                                                            numpy.newaxis]  # without good_rats
            #  ^ todo: DELETE
        sm_likelihood[:, :, s] = u['s'][s] * u['ft|s'][:, :, s]
    qfts = gcm_likelihood * sm_likelihood
    psi = (qfts * p['ft'][:, :, None]) / (
    numpy.sum(qfts, axis=2, keepdims=True) + numpy.spacing(1))  # psi = p^obs(f, t)/q(s|f, t)
    # M
    for s in xrange(S):
        common_factor = psi[:, :, s] / (u['ft|s'][:, :, s] + numpy.spacing(1))  # not sure whether this step is needed
        # update (f|s, z)
        f_scale_factor = common_factor.dot(u['zt|s'][:, :, s].T)
        u['f|sz'][:, s, :] = normalize(u['f|sz'][:, s, :] * f_scale_factor, axis=0)
        # update q(z, t|s)
        zt_scale_factor = u['f|sz'][:, s, :].T.dot(common_factor)
        u['zt|s'][:, :, s] = normalize(u['zt|s'][:, :, s] * zt_scale_factor, axis=(0, 1))
        # u['ft|s'][:, :, s] = numpy.dot(u['f|sz'][:, s, :], u['zt|s'][:, :, s])
        # update q(r|t, s)
        for r in xrange(R):
            u['r|ts'][r, :, s] = numpy.sum(psi[:, :, s] * (p['r'] == r), axis=0)
        u['r|ts'] = normalize(u['r|ts'], axis=0)
        # update q(s)
        u['s'] = normalize(numpy.sum(psi, axis=(0, 1)))
    if store_gains:
        ugain.append(cross_entropy(p['ft'], p['r'], u['ftr']))
    # ^ todo: DELETE
    return u, ugain, -1

# @profile
def fm_ntf(p, S=None, Z=None, em_iterations=10, q_fixed=dict(), good_rats=None, store_gains=defaults.STORE_GAINS, q_init=None):
    F, T, R = p['ftr'].shape
    if q_init is None:
        q = init_ntf(F, T, R, Z, S, q_init)
    else:
        q = copy.deepcopy(q_init)
    if good_rats is None:
        q['good_rats'] = numpy.ones((F, T))
    else:
        q['good_rats'] = good_rats
    s_scale_factor = numpy.zeros_like(q['s'])
    f_scale_factor = numpy.zeros_like(q['f|sz'])
    r_scale_factor = numpy.zeros_like(q['r|ts'])
    zt_scale_factor = numpy.zeros_like(q['zt|s'])
    for s in xrange(S):
        q['ft|s'][:, :, s] = numpy.dot(q['f|sz'][:, s, :], q['zt|s'][:, :, s]) # redundant
        q['ftr'] += q['s'][s]*q['r|ts'][:, :, s].T[numpy.newaxis, :, :]*q['ft|s'][:, :, s, numpy.newaxis]
    if store_gains:
        gain = [cross_entropy(p['ft'], p['r'], q['ftr'])]
    else:
        gain = []
    all_gains = []
    rho = numpy.zeros((F, T, R))

    # new
    u = copy.deepcopy(q)
    psi = copy.deepcopy(rho)
    ugain = copy.deepcopy(gain)
    uall_gains = copy.deepcopy(all_gains)
    for i in xrange(em_iterations):
        sys.stdout.write('*')
        sys.stdout.flush()
        u, ugain, _ = new_updates(u, p, ugain=ugain, uall_gains=uall_gains, S=S, Z=Z, store_gains=store_gains)
        sys.stdout.write('-')
        q, gain, all_gains = old_updates(q, p, gain=gain, all_gains=all_gains, S=S, Z=Z, store_gains=store_gains)
    if defaults.SAVE_FACTORS:
        import pickle
        pickle.dump(u, open('/tmp/vibntf/u.p', 'wb'))
        pickle.dump(q, open('/tmp/vibntf/q.p', 'wb'))
    if defaults.NEW_UPDATES:
        return u, numpy.asarray(ugain), numpy.asarray(uall_gains)
    else:
        return q, numpy.asarray(gain), numpy.asarray(all_gains)


def cross_entropy(pobsft, rft, q):
    assert numpy.all(q >= 0), 'approx distribution not nonnegative!'
    F, T, R = q.shape
    q[q <= 0] = numpy.spacing(1) # avoids nan cross_entropy but maybe breaks monotonic increase of objective
    ce = 0.
    for r in xrange(R):
        if numpy.count_nonzero(r==rft) > 0:
            qft = q[:, :, r]
            ce += numpy.sum(pobsft[rft==r].flatten()*numpy.log(qft[rft==r].flatten()))
    return ce



def calc_gain(p, q):
    R, _, S = q['r|ts'].shape
    qc = copy.deepcopy(q)
    qc['ftr'][:] = 0
    for s in xrange(S):
        qc['ft|s'][:, :, s] = numpy.dot(qc['f|sz'][:, s, :], qc['zt|s'][:, :, s])
        qc['ftr'] += qc['s'][s]*(qc['r|ts'][:, :, s].T[numpy.newaxis, :, :]*qc['ft|s'][:, :, s, numpy.newaxis]*qc['good_rats'][:, :, numpy.newaxis] + (1 - qc['good_rats'][:, :, numpy.newaxis])*(1./R))
    q_ftr = qc['ftr']
    ce = cross_entropy(p['ft'], p['r'], q_ftr)
    return ce


def dense_pftr(p):
    """dense p(f, t, r) from a sparse rep"""
    R = p['R']
    F, T = p['ft'].shape
    p_ftr = numpy.zeros((F, T, R))
    grid = numpy.indices(p['ft'].shape)
    p_ftr[grid[0, :, :].flatten(), grid[1, :, :].flatten(), p['r'].flatten()] = p['ft'].flatten()
    return p_ftr


def get_mask(q):
    q_fts = q['ft|s']*q['s'][numpy.newaxis, numpy.newaxis, :]
    return q_fts/(numpy.sum(q_fts, axis=2)[:, :, numpy.newaxis] + numpy.spacing(1))  # q_sIft


def get_nmf_mask(q):
    q_fts = numpy.dstack([q['f|z'][:, i, None]*q['zt'][i, None, :] for i in xrange(q['zt'].shape[0])])
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
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    for s in range(S):
        fn = 'sep-' + str(s) + '.wav'
        save_sound(sound=y[:, s], sound_fn=save_dir+'/'+fn, fs=fs)
    return y, X_masked