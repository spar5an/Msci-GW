from JHPY import *

config = {
    "mass1" : lambda size: np.random.uniform(10,50,size),
    "mass2" : lambda size: np.random.uniform(10,50,size),
    "spin1z": lambda size: np.random.uniform(-0.5, 0.5, size=size)
}

output = pycbc_data_generator(config, 10000, show_progress=True, num_workers=16, add_noise=True, batch_size=1, time_resolution=1/256)

save_dataloaders(output, "data_no_noise.pt")
