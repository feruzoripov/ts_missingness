---
title: 'ts_missingness: A Python Library for Composable Time-Series Missingness Simulation'
tags:
  - Python
  - time series
  - missing data
  - imputation
  - benchmarking
  - simulation
authors:
  - name: Feruz Oripov
    orcid: 0009-0001-4303-0512
    affiliation: 1
affiliations:
  - name: University of Arizona, USA
    index: 1
date: 10 March 2026
bibliography: paper.bib
---

# Summary

Missing data is pervasive in time-series applications, particularly in healthcare
monitoring, wearable sensors, and environmental sensing, where data loss arises
from device failures, connectivity drops, patient dropout, and sensor degradation.
Evaluating imputation algorithms requires generating controlled missingness in
complete datasets, yet most benchmarking studies rely on simplistic random masking
that fails to capture the structured, temporally correlated missingness observed
in practice.

`ts_missingness` is a Python library that provides composable, reproducible
missingness simulation for time-series data. Its core design contribution is the
explicit separation of *mechanisms* (why data is missing) from *patterns* (how
data is missing) as two orthogonal, independently configurable axes. This enables
researchers to systematically evaluate imputation methods across realistic
combinations---for example, testing whether an algorithm that performs well under
random scattered missingness also handles activity-dependent sensor dropout or
gradual sensor degradation.

# Statement of Need

The missing data literature distinguishes three canonical mechanisms
[@rubin1976inference; @little2019statistical]: Missing Completely At Random
(MCAR), Missing At Random (MAR), and Missing Not At Random (MNAR). In
time-series data, the *temporal structure* of missingness is equally important:
data may be missing as scattered individual points, contiguous blocks (sensor
dropout), monotone tails (participant dropout), gradually increasing gaps (sensor
degradation), or intermittent bursts (flickering connections).

Existing tools address these concerns only partially. The `ampute` function in
the R package `mice` [@vanbuuren2011mice] provides multivariate amputation with
weighted sum scores but lacks temporal pattern awareness and is unavailable in
Python. PyGrinder [@du2023pypots], part of the PyPOTS ecosystem, implements MCAR,
MAR, and MNAR in Python but conflates mechanisms with patterns and does not offer
rate calibration for non-MCAR mechanisms. Most published imputation benchmarks
[@cao2018brits; @du2023saits; @fortuin2020gpvae] use ad-hoc MCAR-only masking
with `numpy.random`, providing no control over temporal structure, no
reproducibility guarantees, and no support for MAR or MNAR evaluation.

`ts_missingness` addresses these gaps by providing:

- **Mechanism--pattern composability**: 3 mechanisms $\times$ 5 patterns = 15
  distinct missingness configurations, all accessible through a single function
  call.
- **Automatic rate calibration**: Binary search offset calibration for MAR and
  MNAR ensures researchers can conduct controlled experiments at specific missing
  rates, not just whatever rate a sigmoid happens to produce.
- **Temporal pattern diversity**: Block, monotone, temporal decay, and Markov
  chain patterns capture real-world missingness structures absent from existing
  Python tools.
- **Weighted multi-driver MAR**: A weighted linear combination of multiple
  observed variables drives missingness probability, enabling realistic
  multi-factor dependency modeling.
- **Native 3D support**: Designed for longitudinal panel data $(N, T, D)$ with
  per-participant normalization, not just single-subject time series.

# Design and Implementation

The library's architecture separates concerns into three modules:

**Mechanisms** (`mechanisms.py`) implement the probabilistic relationship between
data values and missingness. MCAR uses uniform sampling without replacement for
exact rate control. MAR and MNAR use logistic probability models with automatic
offset calibration via binary search to match target missing rates. MAR supports
weighted multi-driver signals computed as
$z_i = \sum_k w_k \cdot (X_{i,k} - \mu_k) / \sigma_k$, where $w_k$ are
user-specified weights.

**Patterns** (`patterns.py`) reshape the temporal structure of the
mechanism-generated mask. Five patterns are implemented:

- *Pointwise*: scattered individual points (identity transform).
- *Block*: contiguous missing segments with configurable length and density.
- *Monotone*: once a dimension goes missing at time $t$, it remains missing for
  all $t' > t$, with dropout times adjusted to match the target count.
- *Temporal decay*: missingness probability increases over time via a sigmoid
  ramp $w(t) = \sigma(\gamma \cdot (t_{\text{norm}} - c))$, with configurable
  steepness and center.
- *Markov chain*: a 2-state Markov chain per series with transition
  probabilities calibrated from the stationary distribution
  $\pi = p_{\text{onset}} / (p_{\text{onset}} + 1 - p_{\text{persist}})$.

**Core API** (`core.py`) composes mechanisms and patterns through a single entry
point:

```python
X_miss, mask = simulate_missingness(
    X,                          # (T, D) or (N, T, D) array
    mechanism="mar",            # WHY: depends on driver
    missing_rate=0.25,          # calibrated to target
    pattern="markov",           # HOW: intermittent bursts
    driver_dims=[0, 1],         # multi-driver
    driver_weights=[0.8, 0.2],  # weighted combination
    persist=0.8,                # Markov stickiness
    seed=42                     # reproducible
)
```

All randomness flows through NumPy's `Generator` API with explicit seed
propagation, ensuring full reproducibility without reliance on global RNG state.

# Comparison with Existing Tools

| Feature | ts\_missingness | PyGrinder | mice::ampute | Ad-hoc scripts |
|---------|:-:|:-:|:-:|:-:|
| MCAR / MAR / MNAR | ✓ | ✓ | ✓ | MCAR only |
| Mechanism--pattern separation | ✓ | ✗ | ✗ | ✗ |
| Block pattern | ✓ | ✗ | ✗ | Rare |
| Monotone pattern | ✓ | ✗ | ✗ | ✗ |
| Temporal decay pattern | ✓ | ✗ | ✗ | ✗ |
| Markov chain pattern | ✓ | ✗ | ✗ | ✗ |
| Rate calibration (MAR/MNAR) | ✓ | ✗ | Partial | ✗ |
| Weighted multi-driver | ✓ | ✗ | ✓ | ✗ |
| 3D $(N, T, D)$ native | ✓ | ✗ | ✗ | ✗ |
| Python | ✓ | ✓ | ✗ (R) | ✓ |
| Reproducible (seeded RNG) | ✓ | ✓ | ✓ | Varies |

# Testing

The library includes 77 automated tests covering all mechanism--pattern
combinations, edge cases (zero/full missing rates, constant signals, pre-existing
NaNs), extreme rate calibration accuracy (1%--90%), numerical stability with
large parameters, input validation, and reproducibility verification. Tests run
in under 0.4 seconds.

# Acknowledgements

This work was conducted as part of a master's thesis at the University of
Arizona.

# References
