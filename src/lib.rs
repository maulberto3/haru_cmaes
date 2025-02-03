//! # CMA-ES Algorithm Implementation
//!
//! This crate provides an implementation of the CMA-ES algorithm.
//!
//! For detailed usage and examples, please refer to `lib.rs` and the examples directory.

pub mod params;

pub mod state;

pub mod strategy;

pub mod fitness;
pub mod objectives;

pub mod utils;

pub mod express_use;

pub mod ask_tell_use;
pub mod fold_use;
