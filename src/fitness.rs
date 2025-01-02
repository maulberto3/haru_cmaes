use anyhow::Result;
use nalgebra::{DMatrix, DVector};

/// Struct to hold the population as normal data points
#[derive(Debug, Clone)]
pub struct PopulationZ {
    pub z: DMatrix<f32>,
}

/// Struct to hold the population as (eigen-)rotated data points
#[derive(Debug, Clone)]
pub struct PopulationY {
    pub y: DMatrix<f32>,
}

#[derive(Debug, Clone)]
pub enum MinOrMax {
    Min,
    Max,
}

/// Structure to hold fitness values of a population.
#[derive(Debug, Clone)]
pub struct Fitness {
    pub values: DVector<f32>,
}

/// A trait for fitness functions.
pub trait FitnessFunction {
    fn cost(&self, pop: &PopulationY) -> DVector<f32>;
    fn cost_dim(&self) -> usize;
    fn optimization_type(&self) -> &MinOrMax;
}

/// A wrapper trait for fitness functions.
pub trait FitnessEvaluator {
    type IndividualsEvaluated;
    type ObjectiveDim;

    fn evaluate(&self, pop: &PopulationY) -> Result<Self::IndividualsEvaluated>;
    fn evaluator_dim(&self) -> Result<Self::ObjectiveDim>;
}

impl<T> FitnessEvaluator for T
where
    T: FitnessFunction,
{
    type IndividualsEvaluated = Fitness;
    type ObjectiveDim = usize;

    fn evaluate(&self, pop: &PopulationY) -> Result<Self::IndividualsEvaluated> {
        let mut values = self.cost(pop);
        let multiplier = match self.optimization_type() {
            MinOrMax::Min => 1.0,
            MinOrMax::Max => -1.0,
        };
        values *= multiplier;
        let fitness = Fitness { values };
        Ok(fitness)
    }

    fn evaluator_dim(&self) -> Result<Self::ObjectiveDim> {
        Ok(self.cost_dim())
    }
}

// /// Function to allow only for complying objective functions
// pub fn allow_objective_func<E: FitnessEvaluator>(
//     objective_function: E,
// ) -> Result<(E, E::ObjectiveDim)> {
//     let objective_dim = objective_function.evaluator_dim()?;
//     Ok((objective_function, objective_dim))
// }
