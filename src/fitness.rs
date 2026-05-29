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

/// A trait for fitness evaluation.
///
/// Implements the full evaluate/evaluator_dim pipeline for any objective function.
pub trait FitnessEvaluator {
    fn evaluate(&self, pop: &PopulationY) -> Result<Fitness>;
    fn evaluator_dim(&self) -> Result<usize>;
}

/// Generic fitness wrapper for user-defined closures.
///
/// `UserFitness` allows users to define custom objective functions as simple closures
/// that take a single individual's vector and return a scalar fitness value.
/// The trait implementation automatically handles mapping over entire populations
/// and applying the optimization direction (min/max).
///
/// # Example
///
/// ```rust
/// use haru_cmaes::fitness::{PopulationY, FitnessEvaluator, UserFitness, MinOrMax};
/// use nalgebra::{Matrix3x4, DMatrix};
///
/// let static_matrix = Matrix3x4::new(
///     1.0, 2.0, 3.0, 3.5,
///     4.0, 5.0, 6.0, 6.5,
///     7.0, 8.0, 9.0, 9.5,
/// );
/// let y = DMatrix::from_row_slice(3, 4, static_matrix.as_slice());
/// let pop = PopulationY { y };
///
/// // Define a custom objective with a closure (sum of squares)
/// let objective_function = UserFitness::new(
///     |individual: &nalgebra::DVector<f32>| {
///         individual.iter().map(|x| x.powi(2)).sum()
///     },
///     4,
///     MinOrMax::Min,
/// );
/// let fitness = objective_function.evaluate(&pop).unwrap();
///
/// // We should have one fitness value per individual
/// assert!(fitness.values.shape() == (3, 1));
/// ```
pub struct UserFitness<F>
where
    F: Fn(&DVector<f32>) -> f32,
{
    func: F,
    obj_dim: usize,
    dir: MinOrMax,
}

impl<F> UserFitness<F>
where
    F: Fn(&DVector<f32>) -> f32,
{
    /// Create a new `UserFitness` objective function.
    ///
    /// # Arguments
    ///
    /// * `func` - A closure that takes an individual (DVector) and returns a fitness value (f32)
    /// * `obj_dim` - The dimension of the objective space
    /// * `dir` - Optimization direction (`MinOrMax::Min` or `MinOrMax::Max`)
    pub fn new(func: F, obj_dim: usize, dir: MinOrMax) -> Self {
        UserFitness { func, obj_dim, dir }
    }
}

impl<F> FitnessEvaluator for UserFitness<F>
where
    F: Fn(&DVector<f32>) -> f32,
{
    fn evaluate(&self, pop: &PopulationY) -> Result<Fitness> {
        let mut values: DVector<f32> = pop
            .y
            .row_iter()
            .map(|row| {
                let row_vec = row.transpose();
                (self.func)(&row_vec)
            })
            .collect::<Vec<f32>>()
            .into();

        let multiplier = match self.dir {
            MinOrMax::Min => 1.0,
            MinOrMax::Max => -1.0,
        };
        values *= multiplier;
        Ok(Fitness { values })
    }

    fn evaluator_dim(&self) -> Result<usize> {
        Ok(self.obj_dim)
    }
}
