import datetime
import shutil
import os
import scipy.signal
import scipy.io
import mir_eval
import numpy
import ntf
import synth
import util
import defaults
import pickle

__author__ = 'ecreager'


class Design(object):
    def __init__(self):
        self.files = True # does the design specify the file paths or store the audio?
        self.fn1 = None
        self.fn2 = None
        self.in1 = None
        self.in2 = None
        self.collection = dict()
        self.snr = None
        self.in1_name = None
        self.in2_name = None

    @staticmethod
    def load_design(design_fn):
        d = pickle.load(open(design_fn, 'rb'))
        return d


class RandomSynthDesign(Design):
    def __init__(self, snr=defaults.MIX_SNR):
        super(Design, self)
        self.files = False
        self.collection = dict({'name':'synth_rand'})
        
        self.p1 = synth.rand_params()
        self.p2 = synth.rand_params()
        self.in1 = synth.fm_square_wave(*self.p1)
        self.in2 = synth.fm_square_wave(*self.p2)
        self.in1_name = synth.get_name_from_params(*self.p1)
        self.in2_name = synth.get_name_from_params(*self.p2)
        self.snr = snr


class RandomSynthAnalysis(Design):
    def __init__(self, snr=defaults.MIX_SNR):
        super(Design, self)
        self.collection = dict({'name':'synth_analysis'})
        self.files = False
        self.p1 = synth.rand_params()
        self.in1 = synth.fm_square_wave(*self.p1)
        self.in2 = ''
        self.in1_name = synth.get_name_from_params(*self.p1)
        self.in2_name = ''
        self.snr = snr
        self.S = 1


class Separation(object):
    def __init__(self):
        pass

    def set_defaults(self):
        self.save_observations = defaults.SAVE_OBS
        self.q_init = None
        self.do_write = defaults.SAVE_AUDIO
        self.fs = defaults.FS
        self.S = defaults.S
        self.Z = defaults.Z
        self.N = defaults.N
        self.hop = defaults.HOP
        self.M = self.N
        self.L = defaults.L
        self.Q = defaults.Q
        self.R = defaults.R
        self.mix_dur = defaults.MIX_DUR
        self.em_iterations = defaults.EM_ITER
        self.q = dict()
        self.p = dict()
        self.bss = dict()
        self.info = dict() #e.g, where to store results
        self.alg_name = self.__unicode__()
        if self.design.files:
            self.mix_name = os.path.basename(self.design.fn1).replace(" ", "-") + '_' + os.path.basename(self.design.fn2).replace(" ", "-") + '_' + str(self.design.snr)
        else:
            self.mix_name = self.design.in1_name + '_' + self.design.in2_name + '_' + str(self.design.snr)
        self.set_info()

    def set_info(self):
        self.info['alg_dir'] = '%s/%s' % (defaults.VIBNTF_DIR, self.alg_name)
        self.info['coll_dir'] = self.info['alg_dir'] + '/' + self.design.collection['name']
        self.info['mix_dir'] = self.info['coll_dir'] + '/' + self.mix_name
        self.info['bss_dir'] = self.info['mix_dir'] + '/bss'
        self.info['in_dir'] = self.info['mix_dir'] + '/in'
        self.info['out_dir'] = self.info['mix_dir'] + '/out'

    def run_separation(self):
        self.setup()
        self.observe(pobs=None) # mix and observe
        self.init_factors()
        self.separate()
        self.recon()
        self.eval()
        if self.save_observations:
            self.save_obs()
        self.cleanup()

    def rerun_separation(self, redo_idx=0, same_init=False):
        self.mix_name += '_redo' + str(redo_idx)
        self.set_info()
        self.setup()
        self.load_obs()
        self.observe(pobs=self.pobs) # mix and observe
        if same_init:
            self.load_init()
        else:
            self.q_init = None
        self.separate(q_init=self.q_init)
        self.recon()
        self.eval()
        if self.save_observations:
            self.save_obs()

    def setup(self):
        pass

    def observe(self, pobs=None):
        pass

    def init_factors(self):
        pass

    def separate(self, q_init=None):
        pass

    def recon(self):
        pass

    def eval(self):
        pass

    def cleanup(self):
        if self.do_write or self.save_observations:
            return
        else:
            shutil.rmtree(self.info['mix_dir'])
            return

    def remove_dirs(self): # assumes empty dirs
        for d in ['in_dir', 'out_dir', 'bss_dir', 'mix_dir']:
            if d == 'bss_dir':
                for f in os.listdir(self.info['bss_dir']):
                    os.remove(self.info[d] + '/' + f)

    def save_obs(self):
        self.design.obs_fn = self.info['in_dir'] + '/pobs.mat'
        self.design_fn = self.info['in_dir'] + '/design.p'
        scipy.io.savemat(self.design.obs_fn, self.pobs)
        self.design.init_fn = self.info['in_dir'] + '/q_init.mat'
        scipy.io.savemat(self.design.init_fn, self.q_init)
        print('\nsaving observations and q_init at %s' % self.info['in_dir'])
        pickle.dump(self.design, open(self.design_fn, 'wb'))

    def load_obs(self):
        self.pobs = scipy.io.loadmat(self.design.obs_fn)

    def load_init(self):
        self.q_init = scipy.io.loadmat(self.design.init_fn, squeeze_me=False)
        self.q_init['s'] = self.q_init['s'].squeeze(axis=0)

    def __unicode__(self):
        return 'dummy'


class VibNtfAnalysis(Separation):
    def __init__(self, design):
        super(Separation, self)
        self.design = design
        self.set_defaults()
        self.S = 1

    def setup(self):
        for d in [self.info['alg_dir'], self.info['coll_dir'], self.info['mix_dir'], self.info['bss_dir'], self.info['in_dir'], self.info['out_dir']]:
            if not os.path.exists(d):
                os.mkdir(d)
        if not self.design.files:
            self.design.fn1 = self.info['in_dir'] + '/in1.wav'

    def observe(self, pobs=None):
        if self.design.files:
            _, self.design.s1 = util.load_sound(self.design.fn1, fs_expected=self.fs)
            if self.design.s1.ndim > 1:
                self.design.s1 = self.design.s1[:, 0]
        else:
            self.design.s1 = self.design.in1
        self.design.s1 = util.center_and_segment(self.design.s1, int(self.mix_dur*self.fs))
        self.mix = self.design.s1
        self.refs = self.design.s1[:, numpy.newaxis]
        if defaults.DO_NOISE:
            noise_var = numpy.var(self.mix)*10**(-defaults.NOISING_SNR/10)
            noise = numpy.sqrt(noise_var)*numpy.random.randn(len(self.mix), )
            self.refs += noise[:, numpy.newaxis]
            self.mix += noise
        self.X, _  = util.ec_stft(x=self.mix, N=self.N, hop=self.hop)
        if pobs is None:
            self.pobs = util.observe(x=self.mix, N=self.N, M=self.M, Q=self.Q, L=self.L, hop=self.hop, fs=self.fs, R=self.R)
        else:
            self.pobs = pobs
        self.R = self.pobs['R']  # in case R becomes odd
        if self.do_write:  # write audio in
            print('\nsaving input mixture at %s' % self.info['in_dir'])
            for s in range(self.S):
                util.save_sound(sound_fn=self.info['in_dir'] + '/in' + str(s) + '.wav', sound=self.refs[:, s], fs=self.fs)
            util.save_sound(sound_fn=self.info['in_dir'] + '/mix.wav', sound=self.mix, fs=self.fs)

    def separate(self, q_init=None):
        if q_init is None:
            F, T = self.pobs['ft'].shape
            R = self.R
            self.q_init = ntf.init_ntf(F, T, R, self.Z, self.S)
        else:
            self.q_init = q_init
        self.q = ntf.fm_ntf(self.pobs, S=self.S, Z=self.Z, em_iterations=self.em_iterations, good_rats=self.pobs['good_rats'], q_init=self.q_init)

    def recon(self):
        self.mask = ntf.get_mask(self.q)
        self.seps, self.X_masked = ntf.apply_mask(X=self.X, mask=self.mask, do_save=self.do_write, save_dir=self.info['out_dir'], hop=self.hop, fs=self.fs)
        self.seps = self.seps[:self.refs.shape[0], :]  # trim
        self.seps = numpy.vstack((self.seps, numpy.zeros((self.refs.shape[0]-self.seps.shape[0], self.S)))) # zero-pad

    def eval(self):
        pass

    def __unicode__(self):
        return 'vibntf'


class VibNtfSeparation(Separation):
    def __init__(self, design):
        super(Separation, self)
        self.design = design
        self.set_defaults()


    def setup(self):
        for d in [self.info['alg_dir'], self.info['coll_dir'], self.info['mix_dir'], self.info['bss_dir'], self.info['in_dir'], self.info['out_dir']]:
            if not os.path.exists(d):
                os.mkdir(d)
        if not self.design.files:
            self.design.fn1 = self.info['in_dir'] + '/in1.wav'
            self.design.fn2 = self.info['in_dir'] + '/in2.wav'

    def separate(self, q_init=None):
        if q_init is None:
            F, T = self.pobs['ft'].shape
            R = self.R
            self.q_init = ntf.init_ntf(F, T, R, self.Z, self.S)
        else:
            self.q_init = q_init
        self.q = ntf.fm_ntf(self.pobs, S=self.S, Z=self.Z, em_iterations=self.em_iterations, good_rats=self.pobs['good_rats'], q_init=self.q_init)

    def observe(self, pobs=None):
        if self.design.files:
            self.mix, self.refs = util.mix_files(self.design.fn1, self.design.fn2, snr=self.design.snr, fs=self.fs, desired_length_sec=self.mix_dur)
        else:
            self.mix, self.refs = util.mix(self.design.in1, self.design.in2, snr=self.design.snr, fs=self.fs, desired_length_sec=self.mix_dur)
        if defaults.DO_NOISE:
            noise_var = numpy.var(self.mix)*10**(-defaults.NOISING_SNR/10)
            noise = numpy.sqrt(noise_var)*numpy.random.randn(len(self.mix), )
            self.refs += noise[:, numpy.newaxis]/2
            self.mix = numpy.sum(self.refs, axis=1)
        self.X, _  = util.ec_stft(x=self.mix, N=self.N, hop=self.hop)
        if pobs is None:
            self.pobs = util.observe(x=self.mix, N=self.N, M=self.M, Q=self.Q, L=self.L, hop=self.hop, fs=self.fs, R=self.R)
        else:
            self.pobs = pobs
        self.R = self.pobs['R']  # in case R becomes odd
        if self.do_write: # write audio in
            print('\nsaving experiment design and q_init at %s' % self.info['in_dir'])
            for s in range(self.S):
                util.save_sound(sound_fn=self.info['in_dir'] + '/in' + str(s) + '.wav', sound=self.refs[:, s], fs=self.fs)
            util.save_sound(sound_fn=self.info['in_dir'] + '/mix.wav', sound=self.mix, fs=self.fs)

    def recon(self):
        self.mask = ntf.get_mask(self.q)
        self.seps, self.X_masked = ntf.apply_mask(X=self.X, mask=self.mask, do_save=self.do_write, save_dir=self.info['out_dir'], hop=self.hop, fs=self.fs)
        self.seps = self.seps[:self.refs.shape[0], :] # trim
        self.seps = numpy.vstack((self.seps, numpy.zeros((self.refs.shape[0]-self.seps.shape[0], self.S)))) # zero-pad

    def eval(self):
        self.bss['sdr'], self.bss['sir'], self.bss['sar'], self.bss['perm'] = mir_eval.separation.bss_eval_sources(self.refs.T, self.seps.T)
        fn_txt = self.info['bss_dir'] + '/bss.txt'
        f = open(fn_txt, 'a+')
        self.eval_txt = 'alg VibNTF\n fn1 %s\n fn2 %s\n snr %s\n fm_ntf bss_eval results\n sdr %s \n sir %s\n sar %s' % (self.design.fn1, self.design.fn2, self.design.snr, self.bss['sdr'], self.bss['sir'], self.bss['sar'])
        f.write(self.eval_txt)
        f.close()

    def __unicode__(self):
        return 'vibntf'


class NmfSeparation(Separation):
    def __init__(self, design):
        super(Separation, self)
        self.design = design
        self.set_defaults()
        self.Z = self.S

    def setup(self):
        for d in [self.info['alg_dir'], self.info['coll_dir'], self.info['mix_dir'], self.info['bss_dir'], self.info['in_dir'], self.info['out_dir']]:
            if not os.path.exists(d):
                os.mkdir(d)
        if not self.design.files:
            self.design.fn1 = self.info['in_dir'] + '/in1.wav'
            self.design.fn2 = self.info['in_dir'] + '/in2.wav'

    def separate(self, q_init=None):
        if q_init is None:
            F, T = self.pobs['ft'].shape
            self.q_init = ntf.init_nmf(F, T, self.Z)
        else:
            self.q_init = q_init
        self.q = ntf.nmf(self.pobs, self.Z, em_iterations=self.em_iterations, q_init=self.q_init)

    def observe(self, pobs):
        if self.design.files:
            self.mix, self.refs = util.mix_files(self.design.fn1, self.design.fn2, snr=self.design.snr, fs=self.fs, desired_length_sec=self.mix_dur)
        else:
            self.mix, self.refs = util.mix(self.design.in1, self.design.in2, snr=self.design.snr, fs=self.fs, desired_length_sec=self.mix_dur)
        if defaults.DO_NOISE:
            noise_var = numpy.var(self.mix)*10**(-defaults.NOISING_SNR/10)
            noise = numpy.sqrt(noise_var)*numpy.random.randn(len(self.mix), )
            # x = x + noise
            self.refs += noise[:, numpy.newaxis]/2
            self.mix = numpy.sum(self.refs, axis=1)
        self.X, _  = util.ec_stft(x=self.mix, N=self.N, hop=self.hop)
        if pobs is None:
            self.pobs = dict()
            self.pobs['ft'] = util.normalize(numpy.abs(self.X))
        else:
            self.pobs = pobs
        if self.do_write: # write audio in
            for s in range(self.S):
                util.save_sound(sound_fn=self.info['in_dir'] + '/in' + str(s) + '.wav', sound=self.refs[:, s], fs=self.fs)
            util.save_sound(sound_fn=self.info['in_dir'] + '/mix.wav', sound=self.mix, fs=self.fs)


    def recon(self):
        self.mask = ntf.get_nmf_mask(self.q)
        self.seps, self.X_masked = ntf.apply_mask(X=self.X, mask=self.mask, do_save=self.do_write, save_dir=self.info['out_dir'], hop=self.hop, fs=self.fs)
        self.seps = self.seps[:self.refs.shape[0], :] # trim
        self.seps = numpy.vstack((self.seps, numpy.zeros((self.refs.shape[0]-self.seps.shape[0], self.S)))) # zero-pad

    def eval(self):
        self.bss['sdr'], self.bss['sir'], self.bss['sar'], self.bss['perm'] = mir_eval.separation.bss_eval_sources(self.refs.T, self.seps.T)
        fn_txt = self.info['bss_dir'] + '/bss.txt'
        f = open(fn_txt, 'a+')
        self.eval_txt = 'alg NMF\n fn1 %s\n fn2 %s\n snr %s\n fm_ntf bss_eval results\n sdr %s \n sir %s\n sar %s' % (self.design.fn1, self.design.fn2, self.design.snr, self.bss['sdr'], self.bss['sir'], self.bss['sar'])
        f.write(self.eval_txt)
        f.close()

    def __unicode__(self):
        return 'nmf'


class NmfAnalysis(Separation):
    def __init__(self, design):
        super(Separation, self)
        self.design = design
        self.set_defaults()
        self.S = 1

    def setup(self):
        for d in [self.info['alg_dir'], self.info['coll_dir'], self.info['mix_dir'], self.info['bss_dir'], self.info['in_dir'], self.info['out_dir']]:
            if not os.path.exists(d):
                os.mkdir(d)
        if not self.design.files:
            self.design.fn1 = self.info['in_dir'] + '/in1.wav'

    def observe(self, pobs):
        if self.design.files:
            _, self.design.s1 = util.load_sound(self.design.fn1, fs_expected=self.fs)
            if self.design.s1.ndim > 1:
                self.design.s1 = self.design.s1[:, 0]
        else:
            self.design.s1 = self.design.in1
        self.design.s1 = util.center_and_segment(self.design.s1, int(self.mix_dur*self.fs))
        self.mix = self.design.s1
        self.refs = self.design.s1[:, numpy.newaxis]
        if defaults.DO_NOISE:
            noise_var = numpy.var(self.mix)*10**(-defaults.NOISING_SNR/10)
            noise = numpy.sqrt(noise_var)*numpy.random.randn(len(self.mix), )
            self.refs += noise[:, numpy.newaxis]
            self.mix += noise
        self.X, _  = util.ec_stft(x=self.mix, N=self.N, hop=self.hop)
        if pobs is None:
            self.pobs = dict()
            self.pobs['ft'] = util.normalize(numpy.abs(self.X))
        else:
            self.pobs = pobs
        if self.do_write: # write audio in
            for s in range(self.S):
                util.save_sound(sound_fn=self.info['in_dir'] + '/in' + str(s) + '.wav', sound=self.refs[:, s], fs=self.fs)
            util.save_sound(sound_fn=self.info['in_dir'] + '/mix.wav', sound=self.mix, fs=self.fs)

    def separate(self, q_init=None):
        self.q = ntf.nmf(self.pobs, self.Z, em_iterations=self.em_iterations, q_init=q_init)

    def recon(self):
        self.mask = ntf.get_nmf_mask(self.q)
        self.seps, self.X_masked = ntf.apply_mask(X=self.X, mask=self.mask, do_save=self.do_write, save_dir=self.info['out_dir'], hop=self.hop, fs=self.fs)
        self.seps = self.seps[:self.refs.shape[0], :] # trim
        self.seps = numpy.vstack((self.seps, numpy.zeros((self.refs.shape[0]-self.seps.shape[0], self.S)))) # zero-pad

    def eval(self):
        pass

    def __unicode__(self):
        return 'nmf'


class CompareSeparation(object):
    def __init__(self, es):
        self.experiments = es
        self.name = 'compare'
        for e in self.experiments:
            self.name += '_' + e.alg_name
        for e in self.experiments:
            e.mix_name += '_' + e.alg_name
            e.alg_name = self.__unicode__()
            e.set_info()
        self.save = defaults.SAVE_COMPARE

    def run_compare(self):
        for e in self.experiments:
            e.run_separation()
        if self.save:
            self.save_compare()

    def save_compare(self):
        fn_txt = self.experiments[0].info['bss_dir'] + '/compare.txt' # store in alg 1's dir by default
        self.compare_results = 'compare: ' + self.__unicode__() + '\n'
        for e in self.experiments:
            self.compare_results += '==========================\n' + e.eval_txt
        f = open(fn_txt, 'a+')
        f.write(self.compare_results)
        f.close()
        print('\nwriting comparison results in %s' % self.experiments[0].info['bss_dir'])

    def __unicode__(self):
        return self.name


class VibNtfSeparationNmfInit(VibNtfSeparation):
    def __init__(self, design):
        super(VibNtfSeparation, self)
        self.design = design
        self.set_defaults()
        self.nmf_experiment = NmfSeparation(design)

    def setup(self):
        for d in [self.info['alg_dir'], self.info['coll_dir'], self.info['mix_dir'], self.info['bss_dir'], self.info['in_dir'], self.info['out_dir']]:
            if not os.path.exists(d):
                os.mkdir(d)
        if not self.design.files:
            self.design.fn1 = self.info['in_dir'] + '/in1.wav'
            self.design.fn2 = self.info['in_dir'] + '/in2.wav'
        self.nmf_experiment.do_write = False
        self.nmf_experiment.setup()
        self.nmf_experiment.observe(pobs=None)
        self.nmf_experiment.separate()
        if defaults.EVAL_NMF_INIT:
            self.nmf_experiment.recon()
            self.nmf_experiment.eval()
        self.nmf_experiment.remove_dirs()

    def separate(self):
        self.q = ntf.fm_ntf(self.pobs, S=self.S, Z=self.Z, em_iterations=self.em_iterations, good_rats=self.pobs['good_rats'], q_init=self.q_init)

    def init_factors(self):
        F, T = self.pobs['ft'].shape
        R = self.R
        self.q_init = ntf.init_ntf(F, T, R, self.Z, self.S)
        for s in range(self.S): # copy nmf factors as ntf init
            for z in range(self.Z):
                self.q_init['f|sz'][:, s, z] = self.nmf_experiment.q['f|z'][:, s]


class Experiment(object):
    def __init__(self):
        pass

    def set_defaults(self):
        self.time_stamp = '{:%Y-%m-%d_%H.%M.%S}'.format(datetime.datetime.now())
        self.results_dir = '%s/%s' % (defaults.VIBNTF_DIR, self.time_stamp)
        self.results = dict()
        self.mean_results = dict()
        for a in self.algs:
            self.results[a] = []

    def setup(self):
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def run_experiment(self):
        self.setup()
        for n in range(self.n_separations):
            print('\nexperiment %i/%i\n' % (n+1, self.n_separations))
            d = self.make_design()
            vibntf = VibNtfSeparation(d)
            nmf = NmfSeparation(d)
            es = [vibntf, nmf]
            # for e in es:
            #     e.save_observations = False
            #     e.do_write = False
            compare = CompareSeparation([vibntf, nmf])
            compare.run_compare()
            self.update_results(compare)
        self.save_results()

    def update_results(self, completed_comparison):
        for e in completed_comparison.experiments:
            self.results[e.__unicode__()].append(e.bss)

    def save_results(self):
        self.results_fn = self.results_dir + '/eval.txt'
        self.latex_fn = self.results_dir + '/table.tex'
        self.results_txt = '%s\nn_separations = %i\nsignal duration (sec): %.1f\nem iterations: %i\n' % (self.name, self.n_separations, defaults.SIG_DUR, defaults.EM_ITER)
        self.results_txt += '----------------------------------------\n'
        for a in self.algs:
            self.mean_results[a] = self.generate_results_stats(self.results[a])
            self.results_txt += self.text_format_alg_results(a, self.mean_results[a])
            self.results_txt += '=========================================\n'
        f = open(self.results_fn, 'a+')
        f.write(self.results_txt)
        f.close()
        self.latex_table = self.latex_format_results(self.algs, self.mean_results)
        f = open(self.latex_fn, 'a+')
        f.write(self.latex_table)
        f.close()
        print('\nresults at ' + self.results_fn)

    @staticmethod
    def text_format_alg_results(alg, mr):
        return 'alg: %s\n bss_eval results\n  sdr %s (+/- %s) \n  sir %s (+/- %s) \n  sar %s (+/- %s)\n' % (alg, mr['sdr'], mr['sdr_std'], mr['sir'], mr['sir_std'], mr['sar'], mr['sar_std'])

    @staticmethod
    def generate_results_stats(separation_results):
        mean_results = dict()
        all_sdr = [d['sdr'] for d in separation_results]
        all_sar = [d['sar'] for d in separation_results]
        all_sir = [d['sir'] for d in separation_results]
        mean_results['sdr'] = numpy.mean(all_sdr)
        mean_results['sdr_std'] = numpy.std(all_sdr)
        mean_results['sar'] = numpy.mean(all_sar)
        mean_results['sar_std'] = numpy.std(all_sar)
        mean_results['sir'] = numpy.mean(all_sir)
        mean_results['sir_std'] = numpy.std(all_sir)
        return mean_results

    @staticmethod
    def latex_format_results(algs, mean_results):
        latex = ''
        alg_names = dict(nmf='2-part KL-NMF', vibntf='Vibrato NTF')
        format_string = '%s&%2.1f~$\pm$~%2.1f&%2.1f~$\pm$~%2.1f&%2.1f~$\pm$~%2.1f\\\\\n'
        for a in algs:
            latex += format_string % (alg_names[a],
                                      mean_results[a]['sdr'], mean_results[a]['sdr_std'],
                                      mean_results[a]['sir'], mean_results[a]['sir_std'],
                                      mean_results[a]['sar'], mean_results[a]['sar_std'])
        return latex


class SynthExperiment(Experiment):
    def __init__(self, n_separations=defaults.N_SEPARATIONS, algs=defaults.ALGS):
        super(Experiment, self)
        self.n_separations = n_separations
        self.algs = algs
        self.name = 'synth experiment'
        self.set_defaults()
        self.make_design = RandomSynthDesign
