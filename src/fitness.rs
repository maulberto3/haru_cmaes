use crate::strategy::PopulationY;
use anyhow::Result;
use nalgebra::DVector;

//////////////
// TODO
// Allow for min or max in obj func
//////////////

/// Structure to hold fitness values of a population.
#[derive(Debug, Clone)]
pub struct Fitness {
    pub values: DVector<f32>,
}

/// A trait for fitness functions.
pub trait FitnessFunction {
    fn cost(&self, pop: &PopulationY) -> DVector<f32>;
    fn cost_dim(&self) -> usize;
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
        let values = self.cost(pop);
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