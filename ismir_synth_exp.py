import experiments

__author__ = 'ecreager'

n_separations = 1
e = experiments.SynthExperiment(n_separations)
e.run_experiment()

# python -m kernprof -l -v ismir_synth_exp.py
