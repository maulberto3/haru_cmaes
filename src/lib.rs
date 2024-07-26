// use anyhow::Result;

#[allow(unused_imports)]
#[allow(clippy::single_component_path_imports)]
use blas_src;
pub mod params;
pub mod state;
pub mod strategy;
// Optional
pub mod fitness;
pub mod simple_use;
pub mod simple_pengowen;

#[cfg(test)]
mod tests {
    use crate::simple_use;

    #[test]
    fn end_to_end_test() {
        // Assuming simple_use::example() returns a Result
        assert!(simple_use::example().is_ok());
    }
}
