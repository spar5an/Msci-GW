# Msci-GW
Code/notes base for Msci project on gravitational waves and machine learning at Imperial College London. This is designed to run inside the pycbc-el8 docker.

At this point in time, just messing around with feature extraction from time series data.

TODO:
- Update to pycbc data
    - Generate lots of data
    - Apply it to h1 and l1
    - make it into torch tensors
    - datasets
    - time parameter?


- Update to pycbc noise


- Data pre-processing
- Dropout + Batch Normalisation
- Transforms -> Neural Spine Flow? Might be worth checking for others
- Hyper parameter search for best transforms/context layers, CNN, LSTM etc