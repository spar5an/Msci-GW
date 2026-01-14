from JHPY import *

config = {
    "mass1" : lambda size: np.random.uniform(10,50,size),
    "mass2" : lambda size: np.random.uniform(10,50,size),
    "spin1z": lambda size: np.random.uniform(-0.5, 0.5, size=size)
}

output = pycbc_data_generator(config, 40000, show_progress=False, num_workers=16, add_noise=False)

save_dataloaders(output, "data_no_noise.pt")
