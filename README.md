# vibntf
Vibrato NTF: source separation by NMF plus frequency modulation cues; http://arxiv.org/abs/1606.00037
The initial commit represents a minimum viable port from the research codebase. It allows you to reproduce the synthetic data experiment from the ISMIR paper, but some desirable features are missing, e.g., fast/sparse tensor updates, visualization on the inferred model parameters, and an experiment that reads audio from an absolute path and performs the separation. I'm working on porting these features to the public codebase so stay tuned.

package dependencies (install via pip): \{numpy, scipy, pickle, matplotlib, mir_eval\}