print("=== minimal_pycbc_test.py ===")

import sys
print("Python executable:", sys.executable)

# Try to import PyCBC
try:
    from pycbc import waveform
    print("PyCBC import OK")
except Exception as e:
    print("FAILED to import PyCBC")
    print("Error:", repr(e))
    raise

import matplotlib.pyplot as plt

# Generate a simple time-domain waveform for a 30-30 solar-mass binary
hp, hc = waveform.get_td_waveform(
    approximant="SEOBNRv4",
    mass1=30.0,   # in solar masses
    mass2=30.0,   # in solar masses
    delta_t=1.0 / 4096,  # sampling interval (s)
    f_lower=20.0         # starting frequency (Hz)
)

plt.plot(hp.sample_times, hp, label = 'Plus Polarisation (hp)')
plt.legend()
plt.savefig("test.png")

print("hp length:", len(hp))
print("hc length:", len(hc))
print("hp sample (first 5 points):", hp.numpy()[:5])
print("hc sample (first 5 points):", hc.numpy()[:5])

print("=== Done minimal_pycbc_test.py ===")
