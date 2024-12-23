# CMAES in Rust

## Motivation

This is my own implementation of the CMA-ES optimization algorithm based in Hansen's purecma python implementation.

## Roadmap

Although functional at this point, the roadmap is to convert this crate to use `ngalgebra` as evidenced in the benchmark: eigen decomposition is faster, nice!. So, expect changes in the short term.

EDIT: I plan to enhance this library as much as possible with ndarray, so no nalgebra for the moment.

## Simple usage example

```
use std::time::Instant;

use crate::{
    fitness::{allow_objective_func, FitnessEvaluator, SquareAndSum},
    CmaesAlgo, CmaesAlgoOptimizer, CmaesParams, CmaesState, CmaesStateLogic,
};
use anyhow::Result;

pub fn example() -> Result<()> {
    // Take start time
    let start = Instant::now();

    // Check allowed objective function
    let obj = allow_objective_func(SquareAndSum)?;

    // Initialize CMA-ES parameters
    let params = CmaesParams {
        // Required
        popsize: 50,
        xstart: vec![0.0; 50],
        sigma: 0.75,
        // Optional (Objective)
        tol: Some(0.0001),
        obj_value: Some(0.0), // This has to make sense for your objective function
        // Optional (Computational)
        zs: Some(0.01),
    };

    // Create a new CMA-ES instance
    let cmaes = CmaesAlgo::new(params)?;

    // Initialize the CMA-ES state
    let mut state = CmaesState::init_state(&cmaes.validated_params)?;

    // Run the CMA-ES algorithm until close to objective value
    let mut step = 0;
    loop {
        // Generate a new population
        let mut pop = cmaes.ask(&mut state)?;

        // Evaluate the fitness of the population
        let mut fitness = obj.evaluate(&pop)?;

        // Update the state with the new population and fitness values
        state = cmaes.tell(state, &mut pop, &mut fitness)?;

        // Are we there yet?
        let obj_value = cmaes.validated_params.obj_value.as_ref();
        let tol = cmaes.validated_params.tol.as_ref();

        if let (Some(obj_value), Some(tol)) = (obj_value, tol) {
            let curr = state.best_y.first().unwrap();
            if (curr - obj_value).abs() < *tol {
                // If we are close to obj_value less than tol, we are there (break)
                break;
            }
        }
        step += 1;
    }
    // Print the average fitness of the best solutions
    println!(
        "Step {} | Fitness: {:+.4?} | Duration p/step: {:.4} secs",
        step,
        &state.best_y.first().unwrap(),
        (start.elapsed().as_micros() as f32) / 1000000.0 / (step as f32)
    );

    Ok(())
}
```

## Requirements for (ndarray and friends): BLAS algebra

I assume you have a clean brand new linux environment, so follow the instructions. You can also refer to the working Github actions, if that helps you better.

### 1) Install Build Tools (GCC)

The `build-essential` package includes the GCC compiler and other necessary tools for building C programs which are needed for low-level C algebra utilities wrapped by rust crates. This is most likely a requirement for BLAS C bindings used by ndarray and friends.

`sudo apt install build-essential`

### 2) Install pkg-config and OpenSSL Development Libraries

If you encounter `OpenSSL` and `pkg-config` related issues during compilation:

`sudo apt install pkg-config libssl-dev`

### 3) Setting Up Rust Dependencies

Ensure the following dependencies are specified in your `Cargo.toml`:

```
[dependencies]
anyhow = { version = "1.0.86" }
rand = { version = "0.8.5" }
ndarray = { version = "0.15", features = ["blas"] }
blas-src = { version = "0.10.0", features = ["openblas"] }
# openblas-src = { version = "0.10.9", features = ["cblas", "system"] }
ndarray-linalg = { version = "0.16", features = ["openblas-system"] }
ndarray-rand = { version = "0.14" }
```

### 4) Installing OpenBLAS

To use `OpenBLAS system-wide` for ndarray and others, install the `libopenblas-dev` package:

`sudo apt install libopenblas-dev`

For Lapack do:

`sudo apt-get install liblapack-dev libblas-dev`

If you want to check where did it got installed `dpkg-query -L libopenblas-dev`

### 5) Additional Tools

Install `cargo-depgraph`, `graphviz`, `cargo machete` and `git cliff` for ci/cd workflow:

```
sudo apt install graphviz
cargo install cargo-depgraph
cargo install cargo-machete
cargo install git-cliff
```

### 6) Git (if needed)

Since it's a fresh ubuntu build, for git:

`git config --global user.name "Your Name"`
`git config --global user.email "your.email@example.com"`

Then, check github key, if `ssh -T git@github.com` says `git@github.com: Permission denied (publickey)`, then, probably the key pair was lost, due to new ubuntu fresh install, so do `ls -al ~/.ssh` and see if you indeed have keys stored. If not, then `ssh-keygen -t ed25519 -C "youremail@example.com"`, `ssh-add`. Then add it to github.com `cat ~/.ssh/ided25519.pub`. Then paste that under Settings, SSH and GPG Keys and that's it.

### 7) Run simple example

cargo run --example simple_use