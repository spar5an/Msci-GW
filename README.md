# Msci-GW
Code/notes base for Msci project on gravitational waves and machine learning at Imperial College London. This is designed to run inside the pycbc-el8 docker.

At this point in time, just messing around with feature extraction from time series data.

TODO:
- Generate simple data -> Done
- Add noise -> Done
- Build NN model for parameter estimation --> Done
- Training loop -> Done
- Testing of trained model --> Done
- Parameter Search for best model --> Done
- TODO FOR NORMALISING FLOWS --> Done
    - Basic model first, just mapping from latent to parameter space, nothing conditional yet --> Done
    - Data structure: Check how to read in the data correctly --> Done
    - Loss function --> Done
    - Full architecture --> DONE BY HAMZA TYVM HAMZA
- REFACTOR JHPY to work with everything --> Done
    - Rewrite dingo_style using functions in jhpy --> Done
    - Update training function --> Done
    - Guassian function from pytorch --> Done
- HPC TODO
    - Learn how to use HPC --> Done
    - Rewrite dingo_npe to be able to be used in HPC --> Done
    - Enable GPU compatitibility and optimisation --> Done (and super fast)
    - Train model on real PyCBC data
    - Update model depending on results

- For training function, get it to revert to best model when scheduler activates --> Done
- Update to pycbc data
- Update to pycbc noise


- Data pre-processing
- Dropout + Batch Normalisation
- Transforms -> Neural Spine Flow? Might be worth checking for others
- Hyper parameter search for best transforms/context layers, CNN, LSTM etc
