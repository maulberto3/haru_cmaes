use crate::strategy::PopulationY;
use anyhow::Result;
use ndarray::{Array2, Axis};

//////////////
// TODO
// Allow for min or max in obj func
//////////////

/// Structure to hold fitness values of a population.
#[derive(Debug, Clone)]
pub struct Fitness {
    pub values: Array2<f32>, // Fitness values for each individual in the population.
}

/// Trait defining the evaluation method for fitness functions.
pub trait FitnessEvaluator {
    type IndividualsEvaluated;
    type ObjectiveDim;

    fn evaluate(&self, pop: &PopulationY) -> Result<Self::IndividualsEvaluated>;
    fn evaluator_dim(&self) -> Result<Self::ObjectiveDim>;
}

impl FitnessEvaluator for SquareAndSum {
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

/// Function to allow only for complying objective functions
pub fn allow_objective_func<E: FitnessEvaluator>(
    objective_function: E,
) -> Result<(E, E::ObjectiveDim)> {
    let objective_dim = objective_function.evaluator_dim()?;
    Ok((objective_function, objective_dim))
}

/// Implementation of the square and sum as simple fitness function.
///
/// Example of a fitness function
///
/// Example
///
/// ```rust
/// use ndarray_rand::RandomExt;
/// use ndarray::Array2;
/// use rand::distributions::Uniform;
/// use haru_cmaes::fitness::SquareAndSum;
/// use haru_cmaes::strategy::PopulationY;;
///
/// let individuals = 10;
/// let objective_function = SquareAndSum { cost_dim: 15 };
/// let shape = (individuals, objective_function.cost_dim);
/// // random population
/// let y = Array2::random(shape, Uniform::new(-1., 1.));
/// let pop = PopulationY { y };
/// let fitness = objective_function.cost(&pop);
///
/// // We should have one fitness value per individual
/// assert!(fitness.shape() == &[individuals, 1]);
/// ```
#[derive(Debug, Clone)]
pub struct SquareAndSum {
    pub cost_dim: usize,
}

impl SquareAndSum {
    // Required method
    pub fn cost(&self, pop: &PopulationY) -> Array2<f32> {
        pop.y
            .map_axis(Axis(1), |row| row.map(|elem| elem.powi(2)).sum())
            .into_shape((pop.y.nrows(), 1))
            .unwrap()
    }

    // Required method
    pub fn cost_dim(&self) -> usize {
        self.cost_dim
    }
}
