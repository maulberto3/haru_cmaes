use anyhow::Result;
use nalgebra::{DMatrix, DVector};

/// Struct to hold the population as standard normal data points
///
/// # Example
/// ```rust
/// use haru_cmaes::fitness::PopulationZ;
/// use nalgebra::{DMatrix, Matrix3x4};
/// let static_matrix = Matrix3x4::new(
///     -0.1, 0.2, 0.3, -0.3,
///     0.4, -0.2, -0.6, 0.6,
///     0.7, -0.8, 0.9, -0.9,
/// );
/// let z = DMatrix::from_row_slice(3, 4, static_matrix.as_slice());
/// let pop_z = PopulationZ { z };
/// assert!(pop_z.z.shape() == (3, 4));
/// ```
#[derive(Debug, Clone)]
pub struct PopulationZ {
    pub z: DMatrix<f32>,
}

/// Struct to hold the population as eigen rotated and scaled data points
///
/// # Example
/// ```rust
/// use haru_cmaes::fitness::PopulationY;
/// use nalgebra::{DMatrix, Matrix3x4};
/// let static_matrix = Matrix3x4::new(
///     1.0, 2.0, 3.0, 3.5,
///     4.0, 5.0, 6.0, 6.5,
///     7.0, 8.0, 9.0, 9.5,
/// );
/// let y = DMatrix::from_row_slice(3, 4, static_matrix.as_slice());
/// let pop_y = PopulationY { y };
/// assert!(pop_y.y.shape() == (3, 4));
/// ```
#[derive(Debug, Clone)]
pub struct PopulationY {
    pub y: DMatrix<f32>,
}

/// Enum to specify optimization direction (minimization or maximization)
#[derive(Debug, Clone, PartialEq)]
pub enum MinOrMax {
    Min,
    Max,
}

/// Structure to hold fitness values of a population
///
/// # Example
/// ```rust
/// use haru_cmaes::fitness::Fitness;
/// use nalgebra::DVector;
/// let fitness_values = DVector::from_vec(vec![0.1, 0.2, 0.3]);
/// let fitness = Fitness { values: fitness_values };
/// assert!(fitness.values.shape() == (3, 1));
/// ```
#[derive(Debug, Clone)]
pub struct Fitness {
    pub values: DVector<f32>,
}

/// A trait for fitness evaluation
///
/// Implements the full evaluate and evaluator_dim methods for any user-defined objective function
pub trait FitnessEvaluator {
    fn evaluate(&self, pop: &PopulationY) -> Result<Fitness>;
    fn evaluator_dim(&self) -> Result<usize>;
}

/// Implement the `FitnessEvaluator` trait for `UserFitness` to allow it to be used as an objective function in the CMA-ES algorithm.
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

/// Generic fitness wrapper for user-defined objective function.
///
/// `UserFitness` allows users to define custom objective functions as simple closures
/// that take a single individual's vector and return a scalar fitness value.
///
/// The trait implementation automatically handles mapping over entire populations
/// and applying the optimization direction (min/max).
///
/// NOTE: Be sure to match the `obj_dim` with the dimension of the individuals in the population for correct evaluation.
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
///         let mut sum = 0.0;
///         for i in 0..individual.len() {
///             sum += individual[i].powi(2);
///         }
///         sum
///     },
///     4, // This needs to match the CMAES algorith initial parameter dimension.
///     MinOrMax::Min,
/// );
/// assert_eq!(objective_function.evaluator_dim().unwrap(), 4);
/// assert_eq!(objective_function.dir, MinOrMax::Min);
/// ```
pub struct UserFitness<F>
where
    F: Fn(&DVector<f32>) -> f32,
{
    pub func: F,
    pub obj_dim: usize,
    pub dir: MinOrMax,
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
