#[allow(unused_imports)]
#[allow(clippy::single_component_path_imports)]
use blas_src;  // necessary for eig
pub mod params;
pub mod state;
pub mod strategy;
pub mod utils;
pub mod fitness;
pub mod simple_use;

// TODO:
// initialize vectors with capacity
// use arrays where possible
// ise nalgebra for better eig
