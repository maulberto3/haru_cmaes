use crate::strategy::PopulationY;
use anyhow::Result;
use nalgebra::DVector;

//////////////
// TODO
// Allow for min or max in obj func
//////////////

//////////////
// TODO
// Decouple example fitness functions with trait
//////////////

/// Structure to hold fitness values of a population.
#[derive(Debug, Clone)]
pub struct Fitness {
    pub values: DVector<f32>, // Fitness values for each individual in the population.
}

/// Trait defining the evaluation method for fitness functions.
pub trait FitnessEvaluator {
    type IndividualsEvaluated;
    type ObjectiveDim;

    fn evaluate(&self, pop: &PopulationY) -> Result<Self::IndividualsEvaluated>;
    fn evaluator_dim(&self) -> Result<Self::ObjectiveDim>;
}

/// Function to allow only for complying objective functions
pub fn allow_objective_func<E: FitnessEvaluator>(
    objective_function: E,
) -> Result<(E, E::ObjectiveDim)> {
    let objective_dim = objective_function.evaluator_dim()?;
    Ok((objective_function, objective_dim))
}