# CMAES in Rust

## Motivation

This is my own implementation of the CMA-ES optimization algorithm based in Hansen's purecma python implementation.

## Roadmap

Now that I have migrated this crate to nalgebra matrix crate (see ## About Backends), current roadmap is to refactor all code for maximum performance.

Also, I plan to provide some interface utilities for users i.e. fmin(...) for quick and easy optimization, and possibly more examples.

## Simple usage example

Please see simple_use.rs

## About Backends

EDIT: I have made a full migration to nalgebra, so now this crate can be use as is i.e. `cargo run`, without backends i.e. OpenBlas. This crate version will bump to 1.0.0 to reflect this breaking change. You can opt to use a backend of your choice. For that, you can follow nalgebra's instructions as it depends on external libraries. Lastly, I have removed the `build.rs` file.

I have started unlinking my project to not use any backend like OpenBlas. However, I am  missing, at least as of now, a pure Rust implementation of eigen decomposition, as without OpenBlas, this crate won't work (I haven't tried other algebra backends).

This crate should run seemlessly without Openblas or any other backend. However, as stated, as I do not have that pure Rust eigen decomposition method yet, OpenBlas is required. This is because ndarray does not have a pure Rust implementation of the required method. However, nalgebra does. 

Ideally, you would just `cargo run` without openblas and it'd work. And if you'd need more power, you'd just do `cargo run --features=openblas`.

But it turns out that `cargo publ` needs to assert that the crate can operate without features; thus that's why I have my temporal build.rs: to include OpenBlas as default and make this crate run well.

As soon as I have my entire cmaes project implemented without OpenBlas i.e. with nalgebra, I have to stick to that build.rs.

### Additional Tools

Install `cargo-depgraph`, `graphviz`, `cargo machete` and `git cliff` for ci/cd workflow:

```
sudo apt install graphviz
cargo install cargo-depgraph
cargo install cargo-machete
cargo install git-cliff
```