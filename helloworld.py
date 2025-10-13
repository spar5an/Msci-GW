import matplotlib.pyplot as plt
from pycbc.waveform import get_td_waveform

for model in ["SEOBNRv4_opt", "IMRPhenomD"]:
    hp, hc = get_td_waveform(approximant=model,
                             mass1=36,
                             mass2=29,
                             delta_t=1.0/4096,
                             f_lower=20)

    plt.plot(hp.sample_times, hp, label=model)

plt.xlabel("Time (s)")
plt.ylabel("Strain")
plt.title("Gravitational Waveform from Binary Black Hole Merger")
plt.legend()
plt.grid()
plt.savefig("test.png")