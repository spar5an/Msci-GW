"""
Quick test to verify dropout layers are now present in DINGOModel
"""
import sys
sys.path.insert(0, '/opt/pycbc/Msci-GW')

from JHPY import DINGOModel
import torch.nn as nn

# Create a simple model
model = DINGOModel(
    data_dim=100,
    param_dim=2,
    num_detectors=2,
    context_dim=64,
    num_flow_layers=4,
    hidden_dim=128,
    multi_detector_mode='concatenate'
)

# Count dropout layers
dropout_count = 0
for name, module in model.named_modules():
    if isinstance(module, nn.Dropout):
        dropout_count += 1
        print(f"Found Dropout layer: {name} (p={module.p})")

print(f"\nTotal Dropout layers found: {dropout_count}")

if dropout_count > 0:
    print("✓ Dropout layers successfully added!")

    # Test setting dropout rate
    print("\nTesting dropout rate modification...")
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.25

    print("After setting dropout_rate=0.25:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            print(f"  {name}: p={module.p}")

    print("\n✓ Dropout modification works correctly!")
else:
    print("✗ No dropout layers found - something went wrong!")
