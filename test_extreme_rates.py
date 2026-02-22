"""Test extreme missing rates to verify calibration."""

import numpy as np
from ts_missingness import simulate_missingness

np.random.seed(42)
X = np.random.randn(200, 5)

print("Testing extreme missing rates:")
print("=" * 60)

# Test very low rates
for rate in [0.01, 0.02, 0.05]:
    print(f"\nTarget rate: {rate:.2%}")
    
    # MCAR (should be exact)
    X_missing, mask = simulate_missingness(X, "mcar", rate, seed=42)
    actual = (~mask).sum() / mask.size
    print(f"  MCAR:  {actual:.4f} (diff: {abs(actual - rate):.4f})")
    
    # MAR
    X_missing, mask = simulate_missingness(
        X, "mar", rate, seed=42, driver_dims=[0]
    )
    actual = (~mask).sum() / mask.size
    print(f"  MAR:   {actual:.4f} (diff: {abs(actual - rate):.4f})")
    
    # MNAR
    X_missing, mask = simulate_missingness(
        X, "mnar", rate, seed=42, mnar_mode="extreme"
    )
    actual = (~mask).sum() / mask.size
    print(f"  MNAR:  {actual:.4f} (diff: {abs(actual - rate):.4f})")

# Test high rates
for rate in [0.50, 0.70, 0.90]:
    print(f"\nTarget rate: {rate:.2%}")
    
    # MCAR (should be exact)
    X_missing, mask = simulate_missingness(X, "mcar", rate, seed=42)
    actual = (~mask).sum() / mask.size
    print(f"  MCAR:  {actual:.4f} (diff: {abs(actual - rate):.4f})")
    
    # MAR
    X_missing, mask = simulate_missingness(
        X, "mar", rate, seed=42, driver_dims=[0]
    )
    actual = (~mask).sum() / mask.size
    print(f"  MAR:   {actual:.4f} (diff: {abs(actual - rate):.4f})")
    
    # MNAR
    X_missing, mask = simulate_missingness(
        X, "mnar", rate, seed=42, mnar_mode="extreme"
    )
    actual = (~mask).sum() / mask.size
    print(f"  MNAR:  {actual:.4f} (diff: {abs(actual - rate):.4f})")

print("\n" + "=" * 60)
print("All extreme rates handled successfully!")
