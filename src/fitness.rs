use anyhow::Result;
use ndarray::{Array2, Axis};
use crate::strategy::PopulationY;

/// Structure to hold fitness values of a population.
#[derive(Debug, Clone)]
pub struct Fitness {
    /// Fitness values for each individual in the population.
    pub values: Array2<f32>,
}

/// Computes the square and sum of each row in the population matrix.
///
/// # Parameters
/// - `pop`: The `PopulationY` containing the matrix of candidate solutions.
///
/// # Returns
/// A `Result` containing a `Fitness` structure with the computed values, or an error if the computation fails.
pub fn square_and_sum(pop: &PopulationY) -> Result<Fitness> {
    let values = pop
        .y
        .map_axis(Axis(1), |row| row.mapv(|elem| elem.powi(2)).sum())
        .view()
        .into_shape((pop.y.nrows(), 1))
        .unwrap()
        .to_owned();
    Ok(Fitness { values })
}

/// Computes the standard deviation of each row in the population matrix.
///
/// # Parameters
/// - `pop`: The `PopulationY` containing the matrix of candidate solutions.
///
/// # Returns
/// A `Result` containing a `Fitness` structure with the computed values, or an error if the computation fails.
pub fn simple_std(pop: &PopulationY) -> Result<Fitness> {
    let values = pop
        .y
        .map_axis(Axis(1), |row| row.std(1.0))
        .view()
        .into_shape((pop.y.nrows(), 1))
        .unwrap()
        .to_owned();
    Ok(Fitness { values })
}

// TODO:
// Implement additional objective functions such as DEA, Rastrigin, etc.
