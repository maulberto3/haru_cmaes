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
// initialize vectors with capacity
// use arrays where possible
// ise nalgebra for better eig
