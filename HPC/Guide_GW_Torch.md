# Running gwosc + pycbc + gwpy + pytorch on the Imperial HPC (CX3)

This guide picks up where your PyCBC guide left off. Same cluster, same
username, same module stack — we just add `gwpy`, `gwosc`, and a
CUDA-enabled `pytorch` into the same virtual environment, and submit a
**GPU** PBS job on CX3 Phase 2.

Assumption: you've already `ssh`'d in with:

```bash
ssh jc5322@login.cx3.hpc.imperial.ac.uk
```

> ⚠️ **Never run heavy compute (training, long PyCBC jobs, etc.) on the
> login node.** The login node is shared and the sysadmins will kill
> anything CPU-heavy. Everything below is either a cheap import test or
> goes via `qsub`.

---

# 1. Fresh module environment

Every fresh login, start clean:

```bash
module purge
module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0
```

Same modules as before — Python 3.11.3 is the one our venv was built
against, so keep it consistent.

---

# 2. Make a new venv for the full GW + Torch stack

We'll keep the old `pycbcenv` untouched and make a second env so if
something breaks you haven't nuked your working PyCBC setup. Call this
one `gwtorchenv`:

```bash
rm -rf ~/venv/gwtorchenv      # only if it already exists
mkdir -p ~/venv
python -m venv ~/venv/gwtorchenv
source ~/venv/gwtorchenv/bin/activate
```

Your prompt should now start with `(gwtorchenv)`.

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

---

# 3. Install pytorch, pycbc, gwpy, gwosc

**Install order matters.** PyTorch ships its own bundled CUDA runtime
inside its wheels, and it pins NumPy to a compatible range. Install it
first, then let the GW stack resolve around it.

```bash
# 1) PyTorch (GPU build, default CUDA from pytorch.org)
python -m pip install torch

# 2) Gravitational-wave stack
python -m pip install pycbc gwpy gwosc
```

This takes a while — PyTorch + all its CUDA libs is ~2–3 GB, and PyCBC
pulls in a chunky scientific stack.

> 💡 If pip complains about NumPy / SciPy version conflicts between
> PyCBC and PyTorch, install PyTorch with an older index:
> `python -m pip install torch --index-url https://download.pytorch.org/whl/cu121`
> and then retry the second line. The stack tends to settle once
> PyTorch is pinned.

Sanity check every package imports (this runs on the **login node**, so
PyTorch will report CUDA = False here — that's normal, the login node
has no GPU):

```bash
python -c "
import sys, torch, pycbc, gwpy, gwosc
print('python :', sys.executable)
print('torch  :', torch.__version__, '| cuda build:', torch.version.cuda)
print('pycbc  :', pycbc.__version__)
print('gwpy   :', gwpy.__version__)
print('gwosc  :', gwosc.__version__)
"
```

Expected output looks roughly like:

```text
python : /rds/general/user/jc5322/home/venv/gwtorchenv/bin/python
torch  : 2.x.y | cuda build: 12.x
pycbc  : 2.x.y
gwpy   : 3.x.y
gwosc  : 0.x.y
```

If any import fails, stop here and paste the error.

---

# 4. Record the venv path for the PBS script

```bash
realpath ~/venv/gwtorchenv
realpath ~/venv/gwtorchenv/bin/activate
```

You should get:

```text
/rds/general/user/jc5322/home/venv/gwtorchenv
/rds/general/user/jc5322/home/venv/gwtorchenv/bin/activate
```

We use this exact path inside the PBS script later.

---

# 5. Test script that uses all four packages

```bash
mkdir -p ~/GWTorchTest
cd ~/GWTorchTest
nano test_gwtorch.py
```

Paste:

```python
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
from gwpy.timeseries import TimeSeries
print("\n[gwpy]")
try:
    ts = TimeSeries.fetch_open_data("H1", gps - 2, gps + 2, cache=True)
    print("  fetched H1 strain:", ts.size, "samples @",
          float(ts.sample_rate), "Hz")
except Exception as e:
    print("  WARNING: fetch_open_data failed (probably no outbound HTTPS"
          " on this compute node):", repr(e))

print("\n=== Done ===")
```

Save & exit (`Ctrl+O`, `Enter`, `Ctrl+X`).

Run it **interactively on the login node** (the venv must still be
active — you should see `(gwtorchenv)` in your prompt):

```bash
python test_gwtorch.py
```

Everything should work except:

- `torch.cuda.is_available()` will be **False** (login node has no GPU)
- The GWpy `fetch_open_data` call should succeed on the login node
  (outbound HTTPS is allowed there)

If any import-level errors appear, fix those before queueing anything.

---

# 6. PBS script — GPU version (the one you actually want)

PyTorch on CPU is pointless; let's request a GPU. On CX3 Phase 2 the
directive is `ngpus=1`. The L40S (Ada Lovelace, 48 GB) is the newer
card and is recommended for AI/ML workloads over the A100.

```bash
nano run_gwtorch.pbs
```

Paste:

```bash
#!/bin/bash
#PBS -N gwtorch_test
#PBS -l walltime=00:20:00
#PBS -l select=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=L40S
#PBS -j oe

# 1) Clean module environment
module purge

# 2) Same modules used to build the venv
module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0

# 3) Activate the venv (FULL PATH from `realpath`)
source /rds/general/user/jc5322/home/venv/gwtorchenv/bin/activate

# 4) Go to the submission directory
cd "$PBS_O_WORKDIR"

echo "=== DEBUG INFO ==="
echo "hostname     : $(hostname)"
echo "pwd          : $(pwd)"
echo "which python : $(which python)"
python -V

echo
echo "=== nvidia-smi ==="
nvidia-smi || echo "nvidia-smi not found — not on a GPU node?"

echo
echo "=== Imports sanity ==="
python -c "
import torch, pycbc, gwpy, gwosc
print('torch :', torch.__version__, '| cuda:', torch.cuda.is_available(),
      '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
print('pycbc :', pycbc.__version__)
print('gwpy  :', gwpy.__version__)
print('gwosc :', gwosc.__version__)
"

echo
echo "=== Run test_gwtorch.py ==="
python test_gwtorch.py

echo
echo "=== Done ==="
```

Save & exit.

A few lines worth understanding:

| Directive | What it does |
|---|---|
| `select=1` | one compute node (chunk) |
| `ncpus=4` | 4 CPU cores on that node |
| `mem=24gb` | 24 GB RAM |
| `ngpus=1` | one GPU |
| `gpu_type=L40S` | ask specifically for an L40S. Drop this line if you're happy with whichever GPU (L40S or A100) is free first — jobs start sooner. |
| `walltime=00:20:00` | job is killed after 20 min |
| `#PBS -j oe` | merge stderr into stdout (one `.o` file) |

**Scale up your real jobs accordingly** — more CPUs, more RAM, longer
walltime. But don't over-ask; the scheduler rewards right-sized jobs
with shorter queue times.

---

# 7. CPU-only PBS script (for quick GW work without torch)

If you ever want to run a gwpy/pycbc job that doesn't need the GPU, just
drop the GPU directive and halve everything:

```bash
#!/bin/bash
#PBS -N gw_cpu_test
#PBS -l walltime=00:20:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -j oe

module purge
module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0
source /rds/general/user/jc5322/home/venv/gwtorchenv/bin/activate
cd "$PBS_O_WORKDIR"

python test_gwtorch.py
```

No GPU → no queue for GPU nodes → usually starts faster.

---

# 8. Submit the job

```bash
cd ~/GWTorchTest
qsub run_gwtorch.pbs
```

PBS prints a job ID, e.g.:

```text
1234567.pbs
```

Check its status:

```bash
qstat -u jc5322
```

The `S` column tells you the state:

- `Q` — queued (waiting for resources)
- `R` — running
- `F` / gone — finished (look for the `.o` file in the submit dir)

GPU jobs on CX3 can sit in `Q` for a while (minutes to an hour,
occasionally more) depending on cluster load. If it's been hours,
`qstat -f <jobid>` shows the full record including any reason it's
blocked.

---

# 9. Inspect the output

Once finished:

```bash
ls
```

You should see:

```text
test_gwtorch.py
run_gwtorch.pbs
run_gwtorch.pbs.o1234567    <-- merged stdout+stderr
```

View it:

```bash
cat run_gwtorch.pbs.o1234567
```

Inside you want to see:

- `nvidia-smi` listing an L40S (or A100) with ~48 GB / ~40 GB memory
- `torch ... | cuda: True | device: NVIDIA L40S` (or A100)
- `1k x 1k matmul on GPU, sum = <some number>`
- PyCBC generated a waveform
- `GW150914 GPS time = 1126259462.4`
- GWpy fetched 4 s of H1 data (or warned about no network — see below)

---

# 10. Gotchas you'll hit eventually

**Compute nodes may not have outbound HTTPS.** `gwpy.fetch_open_data`
and anything downloading from GWOSC at job-time may fail on compute
nodes. The robust pattern is:

1. On the **login node**, inside the venv, pre-download any GWOSC /
   strain data you need into `~/GWTorchTest/data/` (or your RDS project
   space).
2. In the PBS job, load the data from local disk instead of the network.

GWpy respects a cache — if `cache=True` and the file is already on
disk, it won't re-fetch. You can also use `TimeSeries.read(...)` on
pre-downloaded `.gwf` / `.hdf5` files.

**Use `$TMPDIR` for heavy I/O.** Every compute node has a fast local
scratch at `$TMPDIR` that's wiped when the job ends. For training loops
that thrash a dataset, copy the data there first:

```bash
cp -r ~/GWTorchTest/data "$TMPDIR/"
python train.py --data "$TMPDIR/data"
```

This is much faster than hitting `/rds/` on every batch.

**Pin the CUDA-matched torch if you hit version weirdness.** If
`torch.cuda.is_available()` is False inside a GPU job even though
`nvidia-smi` works, the torch wheel's bundled CUDA probably doesn't
match the driver. Reinstall with an explicit index, e.g.:

```bash
python -m pip install --upgrade --force-reinstall \
    torch --index-url https://download.pytorch.org/whl/cu121
```

**`module purge` is not optional.** The login node picks up a few
default modules that quietly set `PYTHONPATH` / `LD_LIBRARY_PATH` and
will shadow your venv's packages if you don't purge first. Always
`module purge` at the top of every PBS script.

**Don't leave `gpu_type=L40S` pinned unless you need it.** On a busy
day, removing that line lets you land on whichever GPU is free first,
and you'll start running sooner. Only pin it if you need the 48 GB of
VRAM on the L40S.

---

# Quick command cheatsheet

```bash
# login
ssh jc5322@login.cx3.hpc.imperial.ac.uk

# enter the env on a fresh shell
module purge && module load tools/prod && module load Python/3.11.3-GCCcore-12.3.0
source ~/venv/gwtorchenv/bin/activate

# submit / monitor / kill
qsub run_gwtorch.pbs
qstat -u jc5322
qstat -f <jobid>        # full detail, useful when queued forever
qdel  <jobid>           # kill
```
