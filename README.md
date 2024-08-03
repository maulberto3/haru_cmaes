# CMAES in Rust

## Motivation

This is my own implementation of the CMA-ES optimization algorithm based in Hansen's purecma python implementation.

This is version 0.3.0 so expect more enhancements and changes along the way.

## Roadmap

Although functional at this point, the roadmap is to convert this crate to use `ngalgebra` as evidenced in the benchmark: eigen decomposition is faster, nice!. So, expect changes in the short term.

Other improvements will follow as well.

Stay tuned.

## Simple usage example

```
use crate::{
    params::CmaesParams, 
    state::CmaesState, 
    strategy::Cmaes
    fitness::square_and_sum, 
    };
use anyhow::Result;

fn example() -> Result<()> {
    let params = CmaesParams {
        popsize: 10,
        xstart: vec![0.0; 10],
        sigma: 0.75,
    };

    let cmaes = Cmaes::new(&params)?;
    let mut state = CmaesState::init_state(&params)?;
    for _i in 0..150 {
        let mut pop = cmaes.ask(&mut state)?;
        let mut fitness = square_and_sum(&pop)?;
        state = cmaes.tell(state, &mut pop, &mut fitness)?;
    }

    println!("Best y: {:+.4?}", &state.best_y);
    println!("Best y (fitness): {:+.4?}", &state.best_y_fit);

    Ok(())
}

fn main() {
    example();
}
```

## Requirements for (ndarray and friends): BLAS algebra

I assume you have a clean brand new linux environment, so follow the following instructions. You can also refer to the working Github actions, if that helps you better.

### 1) Install Build Tools (GCC)

The `build-essential` package includes the GCC compiler and other necessary tools for building C programs whic are needed for low-level C algebra utilities wrapped by rust crates. This is most likely a requirement for BLAS C bindings used by ndarray and friends.

`sudo apt install build-essential`

### 2) Install pkg-config and OpenSSL Development Libraries

If you encounter `OpenSSL` and `pkg-config` related issues during compilation:

`sudo apt install pkg-config libssl-dev`

### 3) Setting Up Rust Dependencies

Ensure the following dependencies are specified in your `Cargo.toml`:

```
anyhow = { version = "1.0.86" }
rand = { version = "0.8.5" }
rayon = { version = "1.10.0" }
ndarray = { version = "0.15", features = ["blas", "rayon"] }
blas-src = { version = "0.10", features = ["openblas"] }
openblas-src = { version = "0.10", features = ["cblas", "system"] }
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

Install `cargo-depgraph` and `graphviz` for dependency visualization:

```
sudo apt install graphviz
cargo install cargo-depgraph
```

### 6) Git (if needed)

Since it's a fresh ubuntu build, for git:

`git config --global user.name "Your Name"`
`git config --global user.email "your.email@example.com"`

Then, check github key, if `ssh -T git@github.com` says `git@github.com: Permission denied (publickey)`, then, probably the key pair was lost, due to new ubuntu fresh install, so do `ls -al ~/.ssh` and see if you indeed have keys stored. If not, then `ssh-keygen -t ed25519 -C "youremail@example.com"`, `ssh-add`. Then add it to github.com `cat ~/.ssh/ided25519.pub`. Then paste that under Settings, SSH and GPG Keys and that's it.

### 7) Run simple example

cargo test --lib