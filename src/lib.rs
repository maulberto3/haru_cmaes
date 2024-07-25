// use anyhow::Result;

#[allow(unused_imports)]
use blas_src;
pub mod fitness;
// use fitness::square_and_sum;

pub mod params;
// use fitness::square_and_sum;
// use params::CmaesParams;

pub mod state;
// use state::CmaesState;

pub mod strategy;
// use strategy::Cmaes;

pub mod simple_use;

#[cfg(test)]
mod tests {
    use crate::simple_use;

    #[test]
    fn end_to_end_test() {
        _ = simple_use::example();
    }
}
