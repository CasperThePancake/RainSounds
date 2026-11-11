# Rain Sounds

Work-in-progress numerical model for rain droplets impacting a three-layer membrane system

(Physics bachelor thesis at KU Leuven)

## Summary of files

- _single_droplet.py_: Main simulation for a single droplet impacting a triple-membrane system
- _single_membrane.py_: Old slower version for a single-membrane system
- _single_membrane_optimizing.py_: Faster version
- _raindrop_distribution_test.py_: Implementation of rain distributions in real-time
- _raindrop_impact_force.py_: Comparison of two graphs for time-dependent impact force
- _raindrop_impact_force_frequency.py_: Same, but in the frequency domain (amplitude + phase)
- _single_membrane_optimizing_majidcomparison.py_: Using parameters from Majid's code to compare results
- _MAJID_single_membrane_one_drop_v5_final.ipynb_: Majid's code in a Jupyter Notebook
- _dispersion.py_: Comparing dispersion curves for different dampening techniques, also RT(f) data fitting
- _single_membrane_anisotropic.py_: Implementation of simple anisotropic tension
- _single_membrane_damping.py_: Implementation of different damping techniques
- _acc_analysis.py_: Analysis code for accelerometer measurements
- _acc_spectro.py_: Comparing accelerometer measurements in spectral form
- _single_membrane_cacherain.py_: Script to generate many audio files for parameters at once, used as cache for live rainfall audio in _raindrop_distribution_test.py_
