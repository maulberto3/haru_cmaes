# CMAES in Rust

## Motivation

This is my own implementation of the CMA-ES optimization algorithm based in Hansen's purecma python implementation.

## Roadmap

Although functional at this point, the roadmap is to convert this crate to use `ngalgebra` as evidenced in the benchmark: eigen decomposition is faster, nice!. So, expect changes in the short term.

EDIT: I plan to enhance this library as much as possible with ndarray, so no nalgebra for the moment.

## Simple usage example

Please see simple_use.rs

## About OpenBlas

I have started unlinking my project to not use any backend like OpenBlas. However, I am  missing, at least as of now, a pure Rust implementation of eigen decomposition, as without OpenBlas, cmaes won't work (I haven't tried any other algebra backend).

I should have that so that the crate would run seemlessly without Openblas or any other backend. However, as stated I do not have it yet, so OpenBlas is required.

Ideally, you would just `cargo run` without openblas and it'd work. And if you'd need more power, you'd just `cargo run --features=openblas`.

But it turns out that `cargo publ` needs to assert that the crate can operate without features; thus that's why I have my temporal build.rs: to include OpenBlas as default and make my project run well. 

As soon as I have my entire cmaes project implemented without OpenBlas, I have to stick to that build.rs

This may be one good reason to migrate matrix calculations to nalgebra as it has both with and without backend implementations for eigen decomposition (just like pengowen's cmaes crate did).

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