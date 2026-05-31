# CMAES in Rust

## Motivation

This is my own Rust implementation of the CMA-ES optimization algorithm based in Hansen's [cmaes](https://github.com/CMA-ES/pycma) python implementation.

## Roadmap

Version +2 has experimented a major refactor. Please refer to CHANGELOG.md for details. For both new and current users, stick to this new +2 version.

I want to implement faer as backend. Also, do thorough benchmarks with openblas and other backends.

Also, I want to exploit paralell capabilities (currently not implemented).

## Usage examples

Please see either examples/simple.rs or examples/fold.rs to see two ways of using this library. Disregard flamegraph fo rnow.

## Features

- **Ask-Tell Interface**: Flexible control over the optimization loop. Generate candidate solutions with `ask()`, evaluate them, and update the algorithm with `tell()`.
- **Fold Mode**: For convenience, use `rollout_fold()` to run the complete optimization loop automatically with a specified number of generations.
- **Minimization and Maximization**: Support for both minimization and maximization problems via the `MinOrMax` enum.
- **Adaptive Covariance Matrix**: Full CMA-ES covariance matrix adaptation with eigendecomposition for effective exploration-exploitation balance.
- **Diagonal Covariance Option**: Optional `only_diag` flag to enforce sparse (diagonal-only) covariance matrices for faster computation in high dimensions.
- **Evolution Paths**: Separate adaptation of step-size (sigma) and covariance matrix using evolution paths for robust convergence.
- **Convergence Detection**: Automatic detection of convergence based on fitness stagnation within a specified tolerance (not considered in fold mode).
- **Customizable Parameters**: Full control over population size, initial guess, step-size, and all CMA-ES adaptation coefficients.

## Parameters

The `CmaesParams` struct allows fine-tuning of the algorithm. Key parameters include:

| Parameter | Type | Description |
|-----------|------|-------------|
| `popsize` | `i32` | Population size (number of candidates per generation) |
| `xstart` | `Vec<f32>` | Initial guess (mean vector); dimension determines problem size |
| `sigma` | `f32` | Step-size (standard deviation); controls exploration radius |
| `num_gens` | `i32` | Number of generations to run (used in fold mode) |
| `tol` | `f32` | Convergence tolerance; stops if fitness change < tol |
| `only_diag` | `bool` | If true, use only diagonal covariance (faster, less adaptive) |
| `mu` | `i32` | Number of parent/elite individuals (auto-computed as popsize/2) |
| `cc` | `f32` | Cumulation constant for rank-one covariance update |
| `cs` | `f32` | Cumulation constant for step-size adaptation |
| `c1` | `f32` | Learning rate for rank-one covariance update |
| `cmu` | `f32` | Learning rate for rank-mu covariance update |
| `damps` | `f32` | Damping factor for step-size adaptation |

**Example: Creating parameters with custom settings**

```rust
use haru_cmaes::params::{CmaesParams, CmaesParamsValidator};

let params = CmaesParams::new()
    .unwrap()
    .set_popsize(30)
    .unwrap()
    .set_xstart(20, 1.0)  // 20-dimensional problem, start at 1.0
    .unwrap()
    .set_sigma(0.5)
    .unwrap()
    .set_tol(0.01)
    .unwrap()
    .set_only_diag(true)
    .unwrap()
    .set_num_gens(100)
    .unwrap();
```

## Objective functions

To get acquantied with how this library expects the format of the user defined objective functions, here are a couple of examples. 

In general, you just need to define your objective function as a Rust closure. 

### Sum of Squares (Sphere Function)

```rust
use haru_cmaes::fitness::{MinOrMax, UserFitness};
use nalgebra::DVector;

let objective_dim = 10;
let objective_function = UserFitness::new(
    |individual: &DVector<f32>| {
        let mut sum = 0.0;
        for i in 0..individual.len() {
            sum += individual[i] * individual[i];
        }
        sum
    },
    objective_dim,
    MinOrMax::Min,
);
```

### Rastrigin Function

```rust
let objective_dim = 10;
let objective_function = UserFitness::new(
    |individual: &DVector<f32>| {
        let n = individual.len() as f32;
        let a = 10.0;
        let mut sum = a * n;
        for i in 0..individual.len() {
            let x = individual[i];
            sum += x * x - a * (2.0 * std::f32::consts::PI * x).cos();
        }
        sum
    },
    objective_dim,
    MinOrMax::Min,
);
```

### Rosenbrock Function

```rust
let objective_dim = 10;
let objective_function = UserFitness::new(
    |individual: &DVector<f32>| {
        let mut sum = 0.0;
        for i in 0..(individual.len() - 1) {
            let x1 = individual[i];
            let x2 = individual[i + 1];
            sum += 100.0 * (x2 - x1 * x1).powi(2) + (1.0 - x1).powi(2);
        }
        sum
    },
    objective_dim,
    MinOrMax::Min,
);
```

## Performance Benchmarks

Benchmarks on local hardware (unplugged laptop) show significant performance improvements:

**Hardware**: ASUS TUF Dash F15 | Intel Core i7-12650H (10 cores, 16 threads) | 24GB RAM | Windows 11 Home WSL

| Implementation | Time per Generation |
|----------------|-------------------|
| **haru_cmaes (Mine)** | **~0.89 ms** ⚡ |
| Pengowen (Rust) | ~3.25 ms |
| Hansen (Python CMA-ES) | ~470 ms |

**Results**: haru_cmaes is **~3.6x faster** than Pengowen's Rust implementation and **~525x faster** than the reference Python implementation (Hansen). This dramatic speedup makes it practical for large-scale optimization and production deployments.

Run benchmarks locally with:
```bash
cargo bench --bench mine
cargo bench --bench pengowen
cargo bench --bench hansen
```

## How to contribute?

You can contribute any way you like. Let's get in touch: https://github.com/maulberto3