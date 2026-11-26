# Rain Sounds

Proof-of-concept numerical model for rain droplets impacting a three-layer membrane system

(Period 1 physics bachelor thesis @ KU Leuven)

## Summary of files

### Parameter analysis

#### raindrop_impact_force.py

Compares two raindrop impact force models in the time domain

#### raindrop_impact_force_frequency.py

Compares two raindrop impact force models in the frequency domain

#### dispersion.py

Plots the phase speed of the membrane waves as function of their frequency, also the complex part of the wavenumber for different damping methods and compared to experimental reverberation time data

### Experimental analysis

#### acc_analysis.py

Uses 60 experimental measurements of raindrop impact acceleration I performed (different heights + different wetness) to analyze them on a bunch of characteristics

#### acc_spectro.py

Script to compare the power spectral density plots for different accelerometer experiment files

### Raindrop model: solver

#### models.py

Main file containing the current best model(s) for raindrop impact simulations on a single membrane, used for our thesis

### Raindrop model: analysis

#### single_membrane_anisotropic.py

Example of using the main model for anisotropic waves in the membrane, showing off its displacement

#### single_membrane_comparespectra.py

Plots and compares the pressure frequency spectra for different tension values

#### single_membrane_maxdisplacement.py

Plots and compares the maximum displacement for different tension values

#### single_membrane_octavebands.py

Plots and compares the 1/3-octave band spectra for different tension values

#### single_membrane_octavebands_majidcomparison.py

Plots and compares the 1/3-octave band spectrum for an experimental audio recording and its exact-parameter simulation with the model

### Rainfall audio

#### single_membrane_cacherain.py

Uses the main model to generate audio files of a raindrop impact for a range of positions, speeds, and radii and their combinations, then saves them to a cache folder

#### raindrop_distribution_test.py

Uses the cached raindrop sounds folder and a mathematical rain model to simulate real-time rain sounds on the membrane

### Old or unfinished

#### single_droplet.py

Old version of the model for raindrop impact simulations on a triple-membrane structure. Could be fixed up and implemented into _models.py_