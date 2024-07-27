use crate::strategy::PopulationY;
use anyhow::Result;
use ndarray::{Array2, Axis};

/// Structure to hold fitness values of a population.
#[derive(Debug, Clone)]
pub struct Fitness {
    /// Fitness values for each individual in the population.
    pub values: Array2<f32>,
}

/// Trait defining the evaluation method for fitness functions.
pub trait FitnessFunction {
    /// Evaluates the fitness of the given population.
    ///
    /// # Arguments
    ///
    /// * `pop` - A reference to the population for which the fitness is evaluated.
    ///
    /// # Returns
    ///
    /// * `Result<Fitness>` containing the evaluated fitness values.
    fn evaluate(&self, pop: &PopulationY) -> Result<Fitness>;
}

/// Implementation of the square and sum fitness function.
pub struct SquareAndSum;

impl FitnessFunction for SquareAndSum {
    fn evaluate(&self, pop: &PopulationY) -> Result<Fitness> {
        let values = pop
            .y
            .map_axis(Axis(1), |row| row.mapv(|elem| elem.powi(2)).sum())
            .view()
            .into_shape((pop.y.nrows(), 1))
            .unwrap()
            .to_owned();
        Ok(Fitness { values })
    }
}

/// Implementation of the square and sum fitness function.
pub struct SimpleStd;

impl FitnessFunction for SimpleStd {
    fn evaluate(&self, pop: &PopulationY) -> Result<Fitness> {
        let values = pop
            .y
            .map_axis(Axis(1), |row| row.std(1.0))
            .view()
            .into_shape((pop.y.nrows(), 1))
            .unwrap()
            .to_owned();
        Ok(Fitness { values })
    }
}

// TODO:
// Implement additional objective functions such as DEA, Rastrigin, etc.
