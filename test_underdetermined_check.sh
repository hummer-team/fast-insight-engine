#!/bin/bash
# Create a temporary test file to check underdetermined systems
cat > /tmp/test_underdet.py << 'PYEOF'
import numpy as np

# Test if 2 samples with Polynomial mode (3 params) causes issues
# Feature matrix: [1, t, t²] for t=0,1
X = np.array([
    [1.0, 0.0, 0.0],  # t=0
    [1.0, 1.0, 1.0]   # t=1
])
y = np.array([1.0, 4.0])

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"X has {X.shape[1]} parameters but only {X.shape[0]} samples")
print(f"System is underdetermined: {X.shape[0]} < {X.shape[1]}")

# Try least squares
try:
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    print(f"NumPy lstsq succeeded: {coeffs}")
except Exception as e:
    print(f"NumPy lstsq failed: {e}")
PYEOF

python3 /tmp/test_underdet.py
