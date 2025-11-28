import subprocess

print("=== check_gpu.py ===")

# Try nvidia-smi (command line)
try:
    out = subprocess.check_output(["nvidia-smi"], encoding="utf-8")
    print("nvidia-smi output:")
    print(out)
except Exception as e:
    print("Failed to run nvidia-smi:", repr(e))

print("=== Done ===")
