
# Fresh login

On your local machine:

```bash
ssh hm2622@login.hpc.ic.ac.uk
```

On the cluster prompt, we’ll start clean each time with:

```bash
module purge
module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0
```

(You’ve already used this Python module, and it exists on your system.)

---

# Delete any old venv and create a new one

We’ll remove the old venv (if it exists) and make a new, clean one.

```bash
rm -rf ~/venv/pycbc-env        # delete old venv if it exists
mkdir -p ~/venv
python -m venv ~/venv/pycbc-env
```

Activate it:

```bash
source ~/venv/pycbc-env/bin/activate
```

Your prompt should now start with something like `(pycbc-env)`.

Upgrade pip and install `pycbc`:

```bash
python -m pip install --upgrade pip
python -m pip install pycbc
```

This will take a bit.

Confirm `pycbc` is installed and from this venv:

```bash
python -c "import pycbc, sys; print('pycbc OK'); print('pycbc file:', pycbc.__file__); print('python exe:', sys.executable)"
```

You should see something like:

```text
pycbc OK
pycbc file: /rds/general/user/hm2622/home/venv/pycbc-env/lib/python3.11/site-packages/pycbc/__init__.py
python exe: /rds/general/user/hm2622/home/venv/pycbc-env/bin/python
```

If this fails, stop here and paste the error.

---

# 📁 Step 2 — Confirm venv path explicitly

Let’s record the exact absolute path so we can use it in the PBS script:

```bash
realpath ~/venv/pycbc-env
realpath ~/venv/pycbc-env/bin/activate
```

You should get:

```text
/rds/general/user/hm2622/home/venv/pycbc-env
/rds/general/user/hm2622/home/venv/pycbc-env/bin/activate
```

We will use this exact path in the PBS script.

---

# Create a test directory and Python script

Make a clean directory just for this test:

```bash
mkdir -p ~/CurrentTest
cd ~/CurrentTest
```

Create `test_pycbc.py`:

```bash
nano test_pycbc.py
```

Paste this:

```python
print("=== Inside test_pycbc.py ===")

import sys
print("Python executable:", sys.executable)

try:
    import pycbc
    print("PyCBC imported OK, version:", pycbc.__version__)
except Exception as e:
    print("FAILED to import PyCBC")
    print("Error:", repr(e))
```

Save & exit (`Ctrl+s`, `Ctrl+X`).

Test it **interactively** (still in the venv):

```bash
python test_pycbc.py
```

Expected output:

- `=== Inside test_pycbc.py ===`
- `Python executable: /rds/general/user/hm2622/home/venv/pycbc-env/bin/python`
- `PyCBC imported OK, version: ...`

If that fails, paste the full output and stop here.

---

# Create a PBS script that uses this venv

Now we write a PBS job script that does exactly what was just done, but now it queues it onto the HPC :)

Still in `~/CurrentTest`:

```bash
nano test_pycbc_job.pbs
```

Paste this exact script (the first 5 lines starting with # are actually code, not just comments):

```bash
#!/bin/bash
#PBS -N test_pycbc
#PBS -l walltime=00:10:00
#PBS -l select=1:ncpus=1:mem=2gb
#PBS -j oe


module purge   # 1) Clean module environment

module load tools/prod   # 2) Load the same modules as used to create the venv
module load Python/3.11.3-GCCcore-12.3.0

source /rds/general/user/hm2622/home/venv/pycbc-env/bin/activate    # 3) Activate the SAME virtual environment where PyCBC is installed
								                                                    #    Use the FULL PATH you got from `realpath`

cd "$PBS_O_WORKDIR"	  # 4) Go to the directory from which qsub was run


echo "=== DEBUG INFO ==="	# 5) Debug information
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "which python: $(which python)"
python -V

echo "=== Try importing pycbc in PBS job ==="
python -c "import pycbc, sys; print('pycbc OK in PBS'); print('pycbc file:', pycbc.__file__); print('python exe:', sys.executable)"

echo "=== Run test_pycbc.py ==="
python test_pycbc.py

echo "=== Done ==="
```

Save & exit (`Ctrl+O`, `Enter`, `Ctrl+X`).

Check files:

```bash
ls
```

You should see:

```text
test_pycbc.py
test_pycbc_job.pbs
```

---

# Submit the job and inspect output

Submit:

```bash
qsub test_pycbc_job.pbs
```

PBS prints a job ID, e.g.:

```text
1195000.pbs-7
```

Wait a bit, then:

```bash
ls
```

You should see something like:

```text
test_pycbc.py
test_pycbc_job.pbs
test_pycbc_job.pbs.o1195000  <--- This is the output
```

# view the output with
'''bash
cat test_pycbc_job.pbs.o1195000
'''
