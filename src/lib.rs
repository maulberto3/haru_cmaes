//! # CMA-ES Algorithm Implementation
//!
//! This crate provides an implementation of the CMA-ES algorithm.
//! 
//! For detailed usage and examples, please refer to `lib.rs` and the examples directory.


#[allow(unused_imports)]
#[allow(clippy::single_component_path_imports)]
use blas_src; // necessary for eig
pub mod fitness;
pub mod params;
pub mod simple_use;
pub mod state;
pub mod strategy;
pub mod utils;

// TODO:
// Check code and refactor more to mutable, mostly state
// initialize vectors with capacity
// use arrays where possible
// use nalgebra for better eig
