print("=== Inside test_gwtorch.py ===")

import sys, socket
print("Python executable:", sys.executable)
print("Hostname:", socket.gethostname())

# --- 1. PyTorch + GPU check ----------------------------------------
import torch
print("\n[torch]")
print("  version      :", torch.__version__)
print("  cuda build   :", torch.version.cuda)
print("  cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  device count :", torch.cuda.device_count())
    print("  device name  :", torch.cuda.get_device_name(0))
    # tiny GPU tensor op, just to prove the GPU actually works
    x = torch.randn(1024, 1024, device="cuda")
    y = (x @ x).sum().item()
    print("  1k x 1k matmul on GPU, sum =", y)
else:
    print("  (no GPU visible — fine on the login node, bad inside a GPU PBS job)")

# --- 2. PyCBC waveform ---------------------------------------------
import pycbc
from pycbc.waveform import get_td_waveform
print("\n[pycbc]")
print("  version:", pycbc.__version__)
hp, hc = get_td_waveform(
    approximant="IMRPhenomD",
    mass1=30.0, mass2=30.0,
    delta_t=1.0 / 4096, f_lower=30.0,
)
print("  generated waveform, length =", len(hp), "samples")

# --- 3. GWOSC: does the CVMFS / network path work? -----------------
from gwosc.datasets import event_gps
print("\n[gwosc]")
gps = event_gps("GW150914")
print("  GW150914 GPS time =", gps)

# --- 4. GWpy: pull a small strand of LIGO open data ----------------
import socket, traceback
socket.setdefaulttimeout(30)  # fail fast if compute node has no outbound HTTPS

from gwpy.timeseries import TimeSeries
print("\n[gwpy]")
gps_f = float(gps)
try:
    ts = TimeSeries.fetch_open_data("H1", gps_f - 2, gps_f + 2, cache=True)
    print("  fetched H1 strain:", ts.size, "samples @",
          ts.sample_rate.to_value("Hz"), "Hz")
except Exception as e:
    print(f"  WARNING: fetch_open_data failed [{type(e).__name__}]: {e}")
    traceback.print_exc()

# --- 5. gw_datagen support deps (pandas, tqdm, scipy) --------------
print("\n[gw_datagen deps]")
import importlib
_required = ["pandas", "tqdm", "scipy"]
_missing = []
for _pkg in _required:
    try:
        _mod = importlib.import_module(_pkg)
        print(f"  {_pkg:7s}: {getattr(_mod, '__version__', '?')}")
    except ImportError as e:
        print(f"  {_pkg:7s}: MISSING ({e})")
        _missing.append(_pkg)
if _missing:
    print(f"  -> install with: pip install {' '.join(_missing)}")

print("\n=== Done ===")