use crate::{
    fitness::FitnessFunction,
    strategy::PopulationY,
};
use nalgebra::DVector;


/// Implementation of the square and sum as fitness function.
///
/// Example of a fitness function
///
/// Example
///
/// ```rust
/// use haru_cmaes::objectives::SquareAndSum;
/// use haru_cmaes::fitness::FitnessEvaluator;
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
/// let fitness = objective_function.evaluate(&pop).unwrap();
///
/// // We should have one fitness value per individual
/// assert!(fitness.values.shape() == (3, 1));
/// ```
pub struct SquareAndSum {
    pub obj_dim: usize,
}

impl FitnessFunction for SquareAndSum {
    // Required method
    fn cost(&self, pop: &PopulationY) -> DVector<f32> {
        pop.y
            .row_iter()
            .map(|row| row.iter().map(|x| x.powi(2)).sum())
            .collect::<Vec<f32>>()
            .into()
    }

    // Required method
    fn cost_dim(&self) -> usize {
        self.obj_dim
    }
}

/// Implementation of the standard deviation and sum as fitness function.
///
/// Example of a fitness function
///
/// Example
///
/// ```rust
/// use haru_cmaes::objectives::StdAndSum;
/// use haru_cmaes::fitness::FitnessEvaluator;
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
/// let fitness = objective_function.evaluate(&pop).unwrap();
///
/// // We should have one fitness value per individual
/// assert!(fitness.values.shape() == (3, 1));
/// ```
pub struct StdAndSum {
    pub obj_dim: usize,
}

impl FitnessFunction for StdAndSum {
    // Required method
    fn cost(&self, pop: &PopulationY) -> DVector<f32> {
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
    fn cost_dim(&self) -> usize {
        self.obj_dim
    }
}

/// Implementation of the Rastrigin function as fitness function.
///
/// Example of a fitness function
///
/// Example
///
/// ```rust
/// use haru_cmaes::objectives::Rastrigin;
/// use haru_cmaes::fitness::FitnessEvaluator;
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
/// let objective_function = Rastrigin { obj_dim: 4 };
/// let fitness = objective_function.evaluate(&pop).unwrap();
///
/// // We should have one fitness value per individual
/// assert!(fitness.values.shape() == (3, 1));
/// ```
pub struct Rastrigin {
    pub obj_dim: usize,
}

impl FitnessFunction for Rastrigin {
    fn cost(&self, pop: &PopulationY) -> DVector<f32> {
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

    fn cost_dim(&self) -> usize {
        self.obj_dim
    }
}