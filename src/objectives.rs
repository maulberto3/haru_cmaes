use crate::fitness::{FitnessFunction, MinOrMax, PopulationY};
use nalgebra::{DMatrix, DVector};

/// Implementation of the square and sum as fitness function.
///
/// ```rust
/// use haru_cmaes::objectives::SquareAndSum;
/// use haru_cmaes::fitness::{PopulationY, FitnessEvaluator};
/// use haru_cmaes::fitness::MinOrMax;
/// use nalgebra::{Matrix3x4, DMatrix};
///
/// let static_matrix = Matrix3x4::new(
///     1.0, 2.0, 3.0, 3.5,
///     4.0, 5.0, 6.0, 6.5,
///     7.0, 8.0, 9.0, 9.5,
/// );
/// let y = DMatrix::from_row_slice(3, 4, static_matrix.as_slice());
/// let pop = PopulationY { y };
/// let objective_function = SquareAndSum { obj_dim: 4, dir: MinOrMax::Min };
/// let fitness = objective_function.evaluate(&pop).unwrap();
///
/// // We should have one fitness value per individual
/// assert!(fitness.values.shape() == (3, 1));
/// ```
pub struct SquareAndSum {
    pub obj_dim: usize,
    pub dir: MinOrMax,
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

    // Required method
    fn optimization_type(&self) -> &MinOrMax {
        &self.dir
    }
}

/// Implementation of the standard deviation and sum as fitness function.
///
/// ```rust
/// use haru_cmaes::objectives::StdAndSum;
/// use haru_cmaes::fitness::{PopulationY, FitnessEvaluator};
/// use haru_cmaes::fitness::MinOrMax;
/// use nalgebra::{Matrix3x4, DMatrix};
///
/// let static_matrix = Matrix3x4::new(
///     1.0, 2.0, 3.0, 3.5,
///     4.0, 5.0, 6.0, 6.5,
///     7.0, 8.0, 9.0, 9.5,
/// );
/// let y = DMatrix::from_row_slice(3, 4, static_matrix.as_slice());
/// let pop = PopulationY { y };
/// let objective_function = StdAndSum { obj_dim: 4, dir: MinOrMax::Min };
/// let fitness = objective_function.evaluate(&pop).unwrap();
///
/// // We should have one fitness value per individual
/// assert!(fitness.values.shape() == (3, 1));
/// ```
pub struct StdAndSum {
    pub obj_dim: usize,
    pub dir: MinOrMax,
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

    // Required method
    fn optimization_type(&self) -> &MinOrMax {
        &self.dir
    }
}

/// Implementation of the Rastrigin function as fitness function.
///
/// ```rust
/// use haru_cmaes::objectives::Rastrigin;
/// use haru_cmaes::fitness::{PopulationY, FitnessEvaluator};
/// use haru_cmaes::fitness::MinOrMax;
/// use nalgebra::{Matrix3x4, DMatrix};
///
/// let static_matrix = Matrix3x4::new(
///     1.0, 2.0, 3.0, 3.5,
///     4.0, 5.0, 6.0, 6.5,
///     7.0, 8.0, 9.0, 9.5,
/// );
/// let y = DMatrix::from_row_slice(3, 4, static_matrix.as_slice());
/// let pop = PopulationY { y };
/// let objective_function = Rastrigin { obj_dim: 4, dir: MinOrMax::Min };
/// let fitness = objective_function.evaluate(&pop).unwrap();
///
/// // We should have one fitness value per individual
/// assert!(fitness.values.shape() == (3, 1));
/// ```
pub struct Rastrigin {
    pub obj_dim: usize,
    pub dir: MinOrMax,
}

impl FitnessFunction for Rastrigin {
    // Required method
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

    // Required method
    fn cost_dim(&self) -> usize {
        self.obj_dim
    }

    // Required method
    fn optimization_type(&self) -> &MinOrMax {
        &self.dir
    }
}

/// Implementation of a (negative) hyperbole with max at 5.0.
///
/// ```rust
/// use haru_cmaes::objectives::XSquare;
/// use haru_cmaes::fitness::{PopulationY, FitnessEvaluator};
/// use haru_cmaes::fitness::MinOrMax;
/// use nalgebra::{Matrix3x4, DMatrix};
///
/// let static_matrix = Matrix3x4::new(
///     1.0, 2.0, 3.0, 3.5,
///     4.0, 5.0, 6.0, 6.5,
///     7.0, 8.0, 9.0, 9.5,
/// );
/// let y = DMatrix::from_row_slice(3, 4, static_matrix.as_slice());
/// let pop = PopulationY { y };
/// let objective_function = XSquare { obj_dim: 4, dir: MinOrMax::Min };
/// let fitness = objective_function.evaluate(&pop).unwrap();
///
/// // We should have one fitness value per individual
/// assert!(fitness.values.shape() == (3, 1));
/// ```
pub struct XSquare {
    pub obj_dim: usize,
    pub dir: MinOrMax,
}

impl FitnessFunction for XSquare {
    // Required method
    fn cost(&self, pop: &PopulationY) -> DVector<f32> {
        let result = pop
            .y
            .row_iter()
            .map(|row| row.iter().map(|x| -x * x).sum())
            .collect::<Vec<f32>>();
        DVector::from_vec(result).add_scalar(5.0)
    }

    // Required method
    fn cost_dim(&self) -> usize {
        self.obj_dim
    }

    // Required method
    fn optimization_type(&self) -> &MinOrMax {
        &self.dir
    }
}

/// Implementation of an optimization problem with constraint
///
/// ```rust
/// use haru_cmaes::objectives::ConstraintProblem;
/// use haru_cmaes::fitness::{PopulationY, FitnessEvaluator};
/// use haru_cmaes::fitness::MinOrMax;
/// use nalgebra::{Matrix3x4, DMatrix};
///
/// let static_matrix = Matrix3x4::new(
///     1.0, 2.0, 3.0, 3.5,
///     4.0, 5.0, 6.0, 6.5,
///     7.0, 8.0, 9.0, 9.5,
/// );
/// let y = DMatrix::from_row_slice(3, 4, static_matrix.as_slice());
/// let pop = PopulationY { y };
/// let objective_function = ConstraintProblem { obj_dim: 4, dir: MinOrMax::Min, target: 0.5 };
/// let fitness = objective_function.evaluate(&pop).unwrap();
///
/// // We should have one fitness value per individual
/// assert!(fitness.values.shape() == (3, 1));
/// ```
pub struct ConstraintProblem {
    pub obj_dim: usize,
    pub dir: MinOrMax,
    pub target: f32,
}

impl FitnessFunction for ConstraintProblem {
    // Required method
    fn cost(&self, pop: &PopulationY) -> DVector<f32> {
        self.objective(pop) - self.constraint_1(pop)
    }

    // Required method
    fn cost_dim(&self) -> usize {
        self.obj_dim
    }

    // Required method
    fn optimization_type(&self) -> &MinOrMax {
        &self.dir
    }
}

impl ConstraintProblem {
    // For a plot check https://www.wolframalpha.com/input?i=abs%28x+-+1.0%29%5E0.5%2C+%280.5x+-+1.0%29%5E2.0+%2B+x%2F5
    pub fn objective(&self, pop: &PopulationY) -> DVector<f32> {
        // Trivial optimization: reach a target value (pointy cone at 1.0)
        pop.y
            .row_iter()
            .map(|row| (row.iter().map(|x| (x - self.target).abs().sqrt()).sum()))
            .collect::<Vec<f32>>()
            .into()
    }

    pub fn constraint_1(&self, pop: &PopulationY) -> DVector<f32> {
        // Trivial constraint: a steeper objective function
        pop.y
            .row_iter()
            .map(|row| {
                row.iter()
                    .map(|x| (0.5 * x - self.target).powi(2) + x / 5.)
                    .sum()
            })
            .collect::<Vec<f32>>()
            .into()
    }
}

/// Implementation of DEA Efficieny Analysis
/// https://www.mdpi.com/2077-0472/14/7/1032
///
/// ```rust
/// use haru_cmaes::objectives::DEAProblem;
/// use haru_cmaes::fitness::{PopulationY, FitnessEvaluator};
/// use haru_cmaes::fitness::MinOrMax;
/// use nalgebra::{Matrix3x4, DMatrix};
///
/// let static_matrix = Matrix3x4::new(
///     1.0, 2.0, 3.0, 3.5,
///     4.0, 5.0, 6.0, 6.5,
///     7.0, 8.0, 9.0, 9.5,
/// );
/// let y = DMatrix::from_row_slice(3, 4, static_matrix.as_slice());
/// let pop = PopulationY { y };
/// let objective_function = DEAProblem {
///     obj_dim: 4,
///     dir: MinOrMax::Min,
///     output_dim: 2,
///     input_dim: 2,
///     data: DMatrix::from_row_slice(3, 4, &vec![
///         1.2, 3.0, 2.0, 2.5,
///         4.0, 5.5, 6.0, 2.5,
///         6.0, 8.5, 7.0, 9.5,
///      ]) };
/// let fitness = objective_function.evaluate(&pop).unwrap();
///
/// // We should have one fitness value per individual
/// assert!(fitness.values.shape() == (3, 1));
/// ```
pub struct DEAProblem {
    // https://www.mdpi.com/2077-0472/14/7/1032
    pub obj_dim: usize,
    pub dir: MinOrMax,
    pub output_dim: usize,
    pub input_dim: usize,
    pub data: DMatrix<f32>,
}

impl FitnessFunction for DEAProblem {
    // Required method
    fn cost(&self, pop: &PopulationY) -> DVector<f32> {
        self.rollout_pop(pop)
    }

    // Required method
    fn cost_dim(&self) -> usize {
        self.output_dim + self.input_dim
    }

    // Required method
    fn optimization_type(&self) -> &MinOrMax {
        &self.dir
    }
}

impl DEAProblem {
    fn num(&self, out_coef: &DVector<f32>, out_data: &DVector<f32>) -> DVector<f32> {
        out_coef.component_mul(out_data)
    }

    fn den(&self, inp_coef: &DVector<f32>, inp_data: &DVector<f32>) -> DVector<f32> {
        inp_coef.component_mul(inp_data)
    }

    fn dea(
        &self,
        out_coef: &DVector<f32>,
        out_data: &DVector<f32>,
        inp_coef: &DVector<f32>,
        inp_data: &DVector<f32>,
    ) -> f32 {
        self.num(out_coef, out_data).sum() / self.den(inp_coef, inp_data).sum()
    }

    fn objective(
        &self,
        out_coef: &DVector<f32>,
        out_data: &DVector<f32>,
        inp_coef: &DVector<f32>,
        inp_data: &DVector<f32>,
    ) -> f32 {
        self.dea(out_coef, out_data, inp_coef, inp_data).sqrt()
    }

    fn constr(&self, inp_coef: &DVector<f32>, inp_data: &DVector<f32>) -> f32 {
        (self.den(inp_coef, inp_data).sum() - 1.0).abs()
    }

    fn others(
        &self,
        out_coef: &DVector<f32>,
        out_data: &DVector<f32>,
        inp_coef: &DVector<f32>,
        inp_data: &DVector<f32>,
    ) -> f32 {
        // To account only for when others' dea surpasses 1.0
        if (self.dea(out_coef, out_data, inp_coef, inp_data) - 1.0) > 0.0 {
            self.dea(out_coef, out_data, inp_coef, inp_data) - 1.0
        } else {
            0.0
        }
    }

    fn rollout_indiv(&self, coef: &DVector<f32>) -> f32 {
        let out_coef = coef.rows(0, self.output_dim).abs().clone_owned();
        let out_data = self.data.columns(0, self.output_dim);

        let inp_coef = coef
            .rows(self.output_dim, self.input_dim)
            .abs()
            .clone_owned();
        let inp_data = self.data.columns(self.output_dim, self.input_dim);

        let res = out_data
            .row_iter()
            .zip(inp_data.row_iter())
            .enumerate()
            .map(|(i, (out_row, inp_row))| {
                let out_row = out_row.transpose();
                let inp_row = inp_row.transpose();
                if i == 0 {
                    self.objective(&out_coef, &out_row, &inp_coef, &inp_row)
                        - self.constr(&inp_coef, &inp_row)
                } else {
                    -self.others(&out_coef, &out_row, &inp_coef, &inp_row)
                    // -self.constr(&inp_coef, &inp_row)
                }
            })
            .sum();
        res
    }

    fn rollout_pop(&self, pop: &PopulationY) -> DVector<f32> {
        let res: Vec<f32> = pop
            .y
            .row_iter()
            .map(|row| self.rollout_indiv(&row.transpose()))
            .collect();
        DVector::from_vec(res)
    }
}
