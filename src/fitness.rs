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

/// Implementation of the square and sum as simple fitness function.
///
/// Example of a fitness function
///
/// Example
///
/// ```rust
/// use haru_cmaes::fitness::SquareAndSum;
/// use haru_cmaes::strategy::PopulationY;;
/// use nalgebra::{Matrix3x4, DMatrix};
///
/// let static_matrix = Matrix3x4::new(
///     1.0, 2.0, 3.0, 3.5,
///     4.0, 5.0, 6.0, 6.5,
///     7.0, 8.0, 9.0, 9.5,
/// );
/// let y = DMatrix::from_row_slice(3, 4, static_matrix.as_slice());
/// let pop = PopulationY { y };
/// let objective_function = SquareAndSum { obj_dim: 4 };
/// let fitness = objective_function.cost(&pop);
///
/// // We should have one fitness value per individual
/// assert!(fitness.shape() == (3, 1));
/// ```
pub struct SquareAndSum {
    pub obj_dim: usize,
}

impl SquareAndSum {
    // Required method
    pub fn cost(&self, pop: &PopulationY) -> DVector<f32> {
        pop.y
            .row_iter()
            .map(|row| row.iter().map(|x| x.powi(2)).sum())
            .collect::<Vec<f32>>()
            .into()
    }

    // Required method
    pub fn cost_dim(&self) -> usize {
        self.obj_dim
    }
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

/// Implementation of the standard deviation and sum as simple fitness function.
///
/// Example of a fitness function
///
/// Example
///
/// ```rust
/// use haru_cmaes::fitness::StdAndSum;
/// use haru_cmaes::strategy::PopulationY;;
/// use nalgebra::{Matrix3x4, DMatrix};
///
/// let static_matrix = Matrix3x4::new(
///     1.0, 2.0, 3.0, 3.5,
///     4.0, 5.0, 6.0, 6.5,
///     7.0, 8.0, 9.0, 9.5,
/// );
/// let y = DMatrix::from_row_slice(3, 4, static_matrix.as_slice());
/// let pop = PopulationY { y };
/// let objective_function = StdAndSum { obj_dim: 4 };
/// let fitness = objective_function.cost(&pop);
///
/// // We should have one fitness value per individual
/// assert!(fitness.shape() == (3, 1));
/// ```
pub struct StdAndSum {
    pub obj_dim: usize,
}

impl StdAndSum {
    // Required method
    pub fn cost(&self, pop: &PopulationY) -> DVector<f32> {
        pop.y
            .row_iter()
            .map(|row| {
                let mean = row.mean();
                let variance =
                    row.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / row.len() as f32;
                variance.sqrt()
            })
            .collect::<Vec<f32>>()
            .into()
    }

    // Required method
    pub fn cost_dim(&self) -> usize {
        self.obj_dim
    }
}

impl FitnessEvaluator for StdAndSum {
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

// TODO
// Implement another example i.e. DEA optimization, Rastrigin, etc.

pub struct Rastrigin {
    pub obj_dim: usize,
}

impl Rastrigin {
    pub fn cost(&self, pop: &PopulationY) -> DVector<f32> {
        let (a, pi) = (10.0, std::f32::consts::PI);
        pop.y
            .row_iter()
            .map(|row| {
                row.iter()
                    .map(|x| x.powi(2) - a * (2.0 * pi * x).cos() + a)
                    .sum()
            })
            .collect::<Vec<f32>>()
            .into()
    }

    pub fn cost_dim(&self) -> usize {
        self.obj_dim
    }
}

impl FitnessEvaluator for Rastrigin {
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
