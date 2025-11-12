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

- Update Model to do parameter prosterior prediction, look at dingo

OK the way this has been done by dingo and most neural posterior estimations, is using a tool called normalising flows. I have spent like 7 hours trying to understand them and im still not there yet. I have included a simple example written by claude before I start work on my own stuff

TODO FOR NORMALISING FLOWS
- Basic model first, just mapping from latent to parameter space, nothing conditional yet
- Data structure: Check how to read in the data correctly
- Transforms required -> Neural Spine Flow? Might be worth checking for others
- Loss function
- Full architecture




- Update to pycbc noise
- Update to pycbc data
- Data pre-processing
- Finished?
