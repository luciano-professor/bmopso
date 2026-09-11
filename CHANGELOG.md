# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.1] - 2026-09-11

### Changed
- Relicensed project from MIT License to Apache License 2.0.
- Updated project metadata, classifiers, license badges, and documentation.

---

## [1.0.0] - 2026-08-28

### Added
- **Core Algorithm (`BMOPSO`)**:
  - Implemented Binary Multi-Objective Particle Swarm Optimization combining:
    - Sigmoid continuous velocity to binary position probability mapping (*Kennedy & Eberhart, 1997*).
    - External Pareto archive maintenance with Adaptive Hypercube Grid partitioning, hypercube fitness-based leader selection, and crowded hypercube capacity pruning (*Coello Coello et al., 2004*).
    - Personal best update with 50/50 random selection for mutually non-dominated positions (*Coello Coello et al., 2004*).
    - Constrained-Dominance Principle handling inequality constraints $g(x) \le 0$ (*Deb, 2002*).
  - Inherits directly from `pymoo.core.algorithm.Algorithm`.

- **Operators Subpackage (`bmopso.operators`)**:
  - `velocity`: Clamped velocity update with dynamic linear inertia decay ($w_{\text{max}} \to w_{\text{min}}$).
  - `sampling`: Numerically stable Sigmoid transform and boolean position sampling.
  - `mutation`: Non-linear decaying mutation / turbulence probability (*Coello Coello et al., 2004*).
  - `pbest`: Personal best replacement with Coello Coello (2004) random coin-flip for incomparable states.

- **Utilities Subpackage (`bmopso.util`)**:
  - `dominance`: Kalyanmoy Deb's constrained Pareto dominance checks and filtering.
  - `grid`: `AdaptiveGrid` hypercube objective space partitioning, hypercube fitness calculation ($10/N$), roulette leader selection, and crowded hypercube pruning.
  - `archive`: Dynamic `NonDominatedArchive` with adaptive hypercube grid maintenance.

- **Benchmark Integration & Examples**:
  - Seamless integration with [`pymoo-binary-problems`](https://github.com/luciano-professor/pymoo-binary-poblems) benchmark suite (`MKP`, `MUBQP`, `MSTSP`, `MOSCP`, `MOFS`).
  - Standalone executable examples for all 5 benchmarks in `examples/` with native `pymoo.visualization.scatter.Scatter` plots.
  - Full pytest test suite in `tests/`.
