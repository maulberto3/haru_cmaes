// use anyhow::Result;

#[allow(unused_imports)]
#[allow(clippy::single_component_path_imports)]
use blas_src;
pub mod fitness;
pub mod params;
pub mod simple_use;
pub mod state;
pub mod strategy;

#[cfg(test)]
mod tests {
    use crate::simple_use;

    #[test]
    fn end_to_end_test() {
        // Assuming simple_use::example() returns a Result
        assert!(simple_use::example().is_ok());
    }
}
