use crate::strategy::PopulationY;
use anyhow::Result;
use ndarray::{Array2, Axis};

/// Structure to hold fitness values of a population.
#[derive(Debug, Clone)]
pub struct Fitness {
    pub values: Array2<f32>, // Fitness values for each individual in the population.
}

/// Trait defining the evaluation method for fitness functions.
pub trait FitnessEvaluator {
    fn evaluate(&self, pop: &PopulationY) -> Result<Fitness>;
}

/// Example of a fitness function
/// Implementation of the square and sum as simple fitness function.
pub struct SquareAndSum;

impl FitnessEvaluator for SquareAndSum {
    fn evaluate(&self, pop: &PopulationY) -> Result<Fitness> {
        let values = pop
            .y
            .map_axis(Axis(1), |row| row.map(|elem| elem.powi(2)).sum())
            // .view()
            .into_shape((pop.y.nrows(), 1))
            .unwrap()
            // .to_owned()
            ;
        Ok(Fitness { values })
    }
}

/// Example of a fitness function
/// Another eImplementation of the square and sum fitness function.
pub struct SimpleStd;

impl FitnessEvaluator for SimpleStd {
    fn evaluate(&self, pop: &PopulationY) -> Result<Fitness> {
        let values = pop
            .y
            .map_axis(Axis(1), |row| row.std(1.0))
            // .view()
            .into_shape((pop.y.nrows(), 1))
            .unwrap()
            // .to_owned()
            ;
        Ok(Fitness { values })
    }
}

/// Check that objective function implements the FitnessEvaluator trait
#[allow(dead_code)]
pub fn allow_objective_func<E: FitnessEvaluator>(evaluator: E) -> Result<E> {
    Ok(evaluator)
}
