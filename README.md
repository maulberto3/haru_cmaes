# Installing Prerequisites

## Install Make Tools

`sudo apt install make`

## Install Build Tools (GCC)

The `build-essential` package includes the GCC compiler and other necessary tools for building C programs.

`sudo apt install build-essential`

## Install pkg-config and OpenSSL Development Libraries

If you encounter `OpenSSL` and `pkg-config` related issues during compilation:

`sudo apt install pkg-config libssl-dev`

## Setting Up Rust Dependencies

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
ndarray-stats = { version = "0.5.1" }
```

## Installing OpenBLAS

To use `OpenBLAS system-wide` for ndarray and others, install the `libopenblas-dev` package:

`sudo apt install libopenblas-dev`

If you want to check where did it got installed `dpkg-query -L libopenblas-dev`

## Additional Tools

Install `cargo-depgraph` and `graphviz` for dependency visualization:

```
sudo apt install graphviz
cargo install cargo-depgraph
```

## For git

Since it's a fresh ubuntu build, for git:

`git config --global user.name "Your Name"`
`git config --global user.email "your.email@example.com"`

Then, check github key, if `ssh -T git@github.com` says `git@github.com: Permission denied (publickey)`, then, probably the key pair was lost, due to new ubuntu fresh install, so do `ls -al ~/.ssh` and see if you indeed have keys stored. If not, then `ssh-keygen -t ed25519 -C "youremail@example.com"`, `ssh-add`. Then add it to github.com `cat ~/.ssh/ided25519.pub`. Then paste that under Settings, SSH and GPG Keys and that's it.


