# Time-Series Missingness Simulation Library

A reproducible Python framework for simulating realistic missingness patterns in time-series data for imputation benchmarking and evaluation.

This library implements statistically grounded missingness mechanisms—**MCAR**, **MAR**, and **MNAR**—with precise or calibrated missing-rate control, support for multivariate and multi-subject time series, optional block (contiguous) dropout, and full reproducibility via seeded random generators.

It is designed for **research-grade evaluation of imputation models**, especially in healthcare and sensor data settings where missingness is structured, correlated, and non-random.

---

## Why this library?

Most imputation benchmarks rely on simplistic random masking that does not reflect real-world data collection processes. In practice:

- Sensors fail during activity
- Devices drop out for contiguous time windows
- Extreme values are more likely to be missing
- Missingness is correlated with other variables

This library provides a **unified, configurable, and reproducible** framework to simulate these patterns while preserving a known ground truth for fair model comparison.

---

## Key Features

- Three standard missingness mechanisms: **MCAR**, **MAR**, **MNAR**
- Optional **block (contiguous) missingness** to simulate sensor dropout
- Supports both:
  - 2D arrays: `(T, D)` (time × features)
  - 3D arrays: `(N, T, D)` (subjects × time × features)
- **Exact** missing-rate control for MCAR  
- **Calibrated** missing-rate control for MAR/MNAR via binary search
- Fully **reproducible** using NumPy’s `Generator` API
- Respects **existing NaNs** in the data
- Returns both:
  - `X_missing` (with NaNs inserted)
  - `mask` (`True = observed`, `False = missing`)

---

## Mask Convention

- `mask == True` → observed value  
- `mask == False` → missing value  

This allows direct evaluation using:

```python
missing_idx = ~mask
error = np.mean((X[missing_idx] - X_imputed[missing_idx])**2)
```

---

## Key Concept: Mechanisms vs. Patterns

This library explicitly separates two fundamental concepts:

**Missingness Mechanisms (WHY data is missing)**  
Describes the probabilistic relationship between missingness and data values:
- **MCAR**: Independent of all data
- **MAR**: Depends on observed variables  
- **MNAR**: Depends on the missing values themselves

**Missingness Patterns (HOW data is missing)**  
Describes the spatial/temporal structure of missingness:
- **Pointwise**: Individual scattered points (default)
- **Block**: Contiguous segments (sensor dropout)

**Combining Mechanisms and Patterns**  
Any mechanism can use any pattern:
- MCAR + pointwise: Random scattered missing
- MCAR + block: Random sensor dropout periods
- MAR + block: Activity-dependent sensor dropout
- MNAR + block: Value-dependent sensor failure

```python
# Explicit API (v2.0+)
X_missing, mask = simulate_missingness(
    X,
    mechanism="mar",      # WHY: depends on driver
    missing_rate=0.25,
    pattern="block",      # HOW: contiguous segments
    driver_dims=[0],
    block_len=10,
    seed=42
)

# Backward compatible API (still works)
X_missing, mask = simulate_missingness(
    X,
    mechanism="mar",
    missing_rate=0.25,
    driver_dims=[0],
    block=True,           # Automatically sets pattern="block"
    block_len=10,
    seed=42
)
```

---

## Installation

```bash
pip install -e .
```

---

## One-Minute Example

```python
import numpy as np
from ts_missingness import simulate_missingness

X = np.random.randn(1000, 6)

# Simple: MCAR with pointwise pattern (default)
X_miss, mask = simulate_missingness(
    X, mechanism="mcar", missing_rate=0.15, seed=42
)

# Advanced: MAR with block pattern (sensor dropout depends on activity)
X_miss, mask = simulate_missingness(
    X, 
    mechanism="mar",      # WHY: depends on driver dimension
    missing_rate=0.25, 
    pattern="block",      # HOW: contiguous segments
    driver_dims=[0],      # Activity level drives missingness
    block_len=10,         # 10-step dropout periods
    seed=42
)
)

print("Actual missing rate:", (~mask).mean())
```

---

## Missingness Mechanisms

### 1) MCAR — Missing Completely At Random

**Definition**  
Missingness is independent of both observed and unobserved data.

**Mathematical model**

$$P(M_{ij} = 1) = \rho$$

where:
- $M_{ij}$ is the missingness indicator for position $(i,j)$
- $\rho$ is the target missing rate

**Implementation**
- Uniform random sampling without replacement
- Exactly $\lfloor n \times \rho \rfloor$ positions are masked among eligible entries
- Guarantees precise missing-rate control

**Use cases**
- Random packet loss
- Uncorrelated sensor glitches
- Transmission errors

---

### 2) MAR — Missing At Random

**Definition**  
Missingness depends on **observed variables**, but **not** on the missing value itself.

**Mathematical model**

$$P(M_{ij} = 1 \mid X) = \sigma(\alpha \cdot z_i + \beta)$$

where:
- $z_i = \frac{\text{driver}_i - \mu}{\sigma}$ (normalized driver signal)
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ (sigmoid function)
- $\alpha$ = strength parameter
- $\beta$ = calibrated offset

**Procedure**
1. Compute a driver signal from specified dimensions
2. Normalize the driver: $z_i = \frac{\text{driver}_i - \mu}{\sigma}$
3. Convert to probabilities using sigmoid: $p_i = \sigma(\alpha \cdot z_i + \beta)$
4. Calibrate offset $\beta$ via binary search to match target missing rate $\rho$
5. Sample $\text{Bernoulli}(p_{ij})$ at eligible positions

**Use cases**
- Sensor failure during high activity
- Dropout correlated with physiological state
- Context-dependent data loss

---

### 3) MNAR — Missing Not At Random

**Definition**  
Missingness depends on the **value itself** (unobserved when missing). This is the most challenging and least identifiable setting.

**Mathematical model**

$$P(M_{ij} = 1 \mid X_{ij}) = \sigma(\alpha \cdot f(z_{ij}) + \beta)$$

where:
- $z_{ij} = \frac{X_{ij} - \mu_j}{\sigma_j}$ (per-dimension normalization)
- $f(z) = \begin{cases} z & \text{if mode="high"} \\ -z & \text{if mode="low"} \\ |z| & \text{if mode="extreme"} \end{cases}$
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ (sigmoid function)
- $\alpha$ = strength parameter
- $\beta$ = calibrated offset

**Procedure**
1. Normalize each dimension independently: $z_{ij} = \frac{X_{ij} - \mu_j}{\sigma_j}$
2. Compute score based on mode: $s_{ij} = f(z_{ij})$
3. Apply sigmoid to obtain probabilities: $p_{ij} = \sigma(\alpha \cdot s_{ij} + \beta)$
4. Calibrate offset $\beta$ to achieve target missing rate $\rho$
5. Sample $\text{Bernoulli}(p_{ij})$

**Use cases**
- Sensor saturation at extremes
- Ceiling/floor effects
- Detection limits

---

## Block Missingness (Optional)

**Purpose**  
Simulates **contiguous dropout periods**, common in real sensor data.

**Behavior**
- Applied as a **post-processing step** on top of MCAR/MAR/MNAR
- Preserves global missing rate
- Increases temporal correlation of missingness

**Parameters**
- `block=True`
- `block_len`: length of each missing segment
- `block_density`: fraction of missingness placed into blocks

**Use cases**
- Battery depletion
- Device removal
- Connectivity loss

---

## Quick Start

```python
# MCAR: 15% random missing
X_miss, mask = simulate_missingness(X, "mcar", missing_rate=0.15, seed=42)

# MAR: 25% missing driven by dimension 0
X_miss, mask = simulate_missingness(
    X, "mar", missing_rate=0.25, seed=42, driver_dims=[0], strength=2.0
)

# MNAR: 10% extreme values missing
X_miss, mask = simulate_missingness(
    X, "mnar", missing_rate=0.10, seed=42, mnar_mode="extreme", strength=2.0
)

# Block missingness
X_miss, mask = simulate_missingness(
    X, "mcar", missing_rate=0.20, seed=42, block=True, block_len=60, block_density=0.7
)
```

---

## Evaluation Example

```python
X_miss, mask = simulate_missingness(X, "mcar", 0.20, seed=42)

X_imputed = your_imputation_method(X_miss)

missing_idx = ~mask
rmse = np.sqrt(np.mean((X[missing_idx] - X_imputed[missing_idx])**2))
mae = np.mean(np.abs(X[missing_idx] - X_imputed[missing_idx]))

print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
```

---

## API

### `simulate_missingness(X, mechanism, missing_rate, seed=None, **kwargs)`

**Parameters**
- `X`: array of shape `(T, D)` or `(N, T, D)`
- `mechanism`: `"mcar"`, `"mar"`, `"mnar"`
- `missing_rate`: float in `[0, 1]`
- `seed`: optional int
- `**kwargs`: mechanism-specific options

**Returns**
- `X_missing`: array with NaNs inserted
- `mask`: boolean array (`True = observed`, `False = missing`)

---

## Reproducibility

- Uses NumPy’s `Generator` API
- No reliance on global RNG state
- Same seed → identical masks

---

## Testing

```bash
pytest ts_missingness/tests/
```

---

## Citation

```bibtex
@software{ts_missingness,
  author = {Feruz Oripov},
  title = {Time-Series Missingness Simulation Library},
  year = {2026},
  url = {https://github.com/feruzoripov/ts_missingness}
}
```

---

## License

MIT