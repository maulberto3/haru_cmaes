use crate::fitness::{PopulationY, PopulationZ};
// use crate::utils::median;
use crate::{
    fitness::Fitness,
    params::CmaesParams,
    state::{CmaesState, CmaesStateLogic},
};
use anyhow::Result;
use nalgebra::{DMatrix, DVector};
use rand::prelude::Distribution;
use statrs::distribution::MultivariateNormal;

/// Struct to hold the algorithm's data and ask and tell methods
#[derive(Debug)]
pub struct CmaesAlgo {
    pub params: CmaesParams,
}

/// Implementing initial logic for CMA-ES algorithm.
impl CmaesAlgo {
    /// Creates a new CMA-ES algorithm instance.
    ///
    /// ```rust
    /// use haru_cmaes::params::{CmaesParams, CmaesParamsValidator};
    /// use haru_cmaes::strategy::CmaesAlgo;
    ///
    /// let params = CmaesParams::new().unwrap();
    /// let cmaes = CmaesAlgo::new(params);
    ///
    /// assert!(cmaes.is_ok());
    /// ```
    /// doctest this
    ///
    pub fn new(params: CmaesParams) -> Result<Self> {
        // let params = CmaesParams::validate(params)?;
        Ok(Self { params })
    }

    /// Generates a matrix of standard normal random variables.
    ///
    /// ```rust
    /// use haru_cmaes::params::{CmaesParams, CmaesParamsValidator};
    /// use haru_cmaes::strategy::{CmaesAlgo, CmaesAlgoOptimizer};
    /// use haru_cmaes::state::{CmaesState, CmaesStateLogic};
    ///
    /// let params = CmaesParams::new().unwrap();
    /// let cmaes = CmaesAlgo::new(params).unwrap();
    /// let mut state = CmaesState::init_state(&cmaes.params).unwrap();
    /// let z = cmaes.ask_z(&mut state);
    ///
    /// assert!(z.is_ok());
    /// ```
    pub fn ask_z(&self, state: &mut CmaesState) -> Result<PopulationZ> {
        // refactor!
        let mean = DVector::from_vec(vec![0.0; self.params.xstart.len()]);
        let cov = DMatrix::identity(self.params.xstart.len(), self.params.xstart.len());
        let normal = MultivariateNormal::new_from_nalgebra(mean, cov).unwrap();
        let mut rng = rand::thread_rng();
        let mut data: Vec<f32> =
            Vec::with_capacity(self.params.popsize as usize * self.params.xstart.len());
        for _ in 0..self.params.popsize {
            let sample = normal
                .sample(&mut rng)
                .iter()
                .map(|x| *x as f32)
                .collect::<Vec<f32>>(); // Generate a random sample``
            data.extend(sample);
        }
        let z = DMatrix::from_row_slice(
            self.params.popsize as usize,
            self.params.xstart.len(),
            &data,
        );
        state.z.copy_from(&z);
        Ok(PopulationZ { z })
    }
}

/// Trait for CMA-ES algorithm.
pub trait CmaesAlgoOptimizer {
    type NewPopulation;
    type NewState;
    type Done;

    fn ask(&self, state: &mut CmaesState) -> Result<Self::NewPopulation>;
    fn tell(
        &self,
        state: CmaesState,
        pop: &mut PopulationY,
        fitness: &mut Fitness,
    ) -> Result<Self::NewState>;
    fn is_done(&self, state: &CmaesState, step: i32) -> Result<Self::Done>;
}

/// Implementing Trait for CMA-ES algorithm.
impl CmaesAlgoOptimizer for CmaesAlgo {
    type NewPopulation = PopulationY;
    type NewState = CmaesState;
    type Done = bool;

    /// ASK
    /// Generates a new population.
    ///
    /// ```rust
    /// use haru_cmaes::params::{CmaesParams, CmaesParamsValidator};
    /// use haru_cmaes::strategy::{CmaesAlgo, CmaesAlgoOptimizer};
    /// use haru_cmaes::state::{CmaesState, CmaesStateLogic};
    ///
    /// let params = CmaesParams::new().unwrap();
    /// let cmaes = CmaesAlgo::new(params).unwrap();
    /// let mut state = CmaesState::init_state(&cmaes.params).unwrap();
    /// let y = cmaes.ask(&mut state);
    ///
    /// assert!(y.is_ok());
    /// ```
    fn ask(&self, state: &mut CmaesState) -> Result<Self::NewPopulation> {
        #[cfg(feature = "profile_memory")]
        {
            use crate::utils::{format_number, get_memory_usage};
            println!("Memory usage: {} Mb", format_number(get_memory_usage()?));
        }

        state.prepare_ask(&self.params)?;

        let z: DMatrix<f32> = self.ask_z(state)?.z;

        // let eig_vals_sqrt: DMatrix<f32> = DMatrix::from_diag(&state.eig_vals.mapv(f32::sqrt));
        let eig_vals_sqrt: DMatrix<f32> = DMatrix::from_diagonal(
            &state
                .eig_vals
                .iter()
                .map(|x| x.sqrt())
                .collect::<Vec<f32>>()
                .into(),
        );

        let scaled_z: DMatrix<f32> = z.map(|x| x * state.sigma) * &eig_vals_sqrt;
        let rotated_z: DMatrix<f32> = scaled_z * &state.eig_vecs.transpose();
        let y: DMatrix<f32> = DMatrix::from_rows(
            &rotated_z
                .row_iter()
                .map(|row| row + &state.mean.transpose())
                .collect::<Vec<_>>(),
        );

        state.y.copy_from(&y);

        Ok(PopulationY { y })
    }

    /// TELL
    /// Updates the CMA-ES state based on the new population and fitness values.
    ///
    /// ```rust
    /// use haru_cmaes::params::{CmaesParams, CmaesParamsValidator};
    /// use haru_cmaes::strategy::{CmaesAlgo, CmaesAlgoOptimizer};
    /// use haru_cmaes::state::{CmaesState, CmaesStateLogic};
    /// use haru_cmaes::fitness::{FitnessEvaluator, MinOrMax};
    /// use haru_cmaes::objectives::SquareAndSum;
    ///
    /// let params = CmaesParams::new().unwrap();
    /// let cmaes = CmaesAlgo::new(params).unwrap();
    ///
    /// let mut state = CmaesState::init_state(&cmaes.params).unwrap();
    ///
    /// let mut y = cmaes.ask(&mut state).unwrap();
    ///
    /// let obj_func = SquareAndSum {
    ///     obj_dim: 5,
    ///     dir: MinOrMax::Min,
    /// };
    ///
    /// let mut fitness = obj_func.evaluate(&y).unwrap();
    /// // for soem reason, cargo test --doc didn't like without 'let'
    /// let state = cmaes.tell(state, &mut y, &mut fitness);
    ///
    /// assert!(state.is_ok());
    /// ```
    fn tell(
        &self,
        mut state: CmaesState,
        pop: &mut PopulationY,
        fitness: &mut Fitness,
    ) -> Result<Self::NewState> {
        // Init data
        state.g += 1;
        state.evals_count += fitness.values.nrows() as i32;
        let xold = state.mean.to_owned();

        // Sort fitness values and population
        let mut indices: Vec<usize> = (0..fitness.values.nrows()).collect(); // refactor
        indices.sort_by(|&i, &j| {
            fitness.values[(i, 0)]
                .partial_cmp(&fitness.values[(j, 0)])
                .unwrap()
        });
        let sorted_xs: DMatrix<f32> =
            DMatrix::from_rows(&indices.iter().map(|&i| pop.y.row(i)).collect::<Vec<_>>());

        let sorted_fit: DVector<f32> = DVector::from_rows(
            &indices
                .iter()
                .map(|&i| fitness.values.row(i))
                .collect::<Vec<_>>(),
        );
        pop.y.copy_from(&sorted_xs);
        fitness.values.copy_from(&sorted_fit);

        // Record current best solution, update best solution if any
        // println!("{}", &fitness.values);
        // println!("{}", &pop.y);
        state.best_y_hist.push(fitness.values.rows(0, 2).mean());
        if fitness.values[0] < state.best_y_fit[0] {
            state.best_y.copy_from(&pop.y.row(0).transpose());
            state.best_y_fit.copy_from(&fitness.values.row(0));
        }

        // Update mean
        let y_mu: DMatrix<f32> = pop.y.rows(0, self.params.mu as usize).into();
        let weights_mu: DVector<f32> = self.params.weights.rows(0, self.params.mu as usize).into(); // refactor as weights seems to not change at all, already in params
        let y_w: DVector<f32> = y_mu.transpose() * weights_mu;
        state.mean.copy_from(&y_w); // refactor less allocation

        // Update evolution path ps
        let new_y: DVector<f32> = &state.mean - &xold;
        let new_z: DVector<f32> = &state.inv_sqrt * &new_y; // refactor
        let csn = (self.params.cs * (2. - self.params.cs) * self.params.mueff).sqrt() / state.sigma;
        let new_ps = &state.ps * (1. - self.params.cs) + csn * new_z;
        state.ps.copy_from(&new_ps);

        // Update evolution path covariance
        let ccn = (self.params.cc * (2. - self.params.cc) * self.params.mueff).sqrt() / state.sigma;
        let hsig = state.ps.map(|x| x * x).sum()
            / (state.ps.len() as f32)
            / (1. - (1. - self.params.cs).powi(2 * state.evals_count / self.params.popsize));
        let new_pc = &state.pc * (1. - self.params.cs) + ccn * hsig * &new_y;
        state.pc.copy_from(&new_pc);

        // Adapt covariance matrix C
        let c1a =
            self.params.c1 * (1. - (1. - hsig * hsig) * self.params.cc * (2. - self.params.cc));
        state
            .cov
            .copy_from(&state.cov.map(|x| x * (1. - c1a - self.params.cmu)));
        let pc_outer: DMatrix<f32> = &state.pc * &state.pc.transpose().map(|x| x * self.params.c1);
        state.cov = &state.cov + pc_outer;

        // Perform the rank-mu update
        for (i, w) in self.params.weights.iter().enumerate() {
            let mut w = *w;
            if w < 0. {
                w = 0.001
            }
            let dx: DVector<f32> = &pop.y.rows(i, 1).transpose() - &xold;
            let dx: DMatrix<f32> = &dx * &dx.transpose();
            let dx: DMatrix<f32> =
                dx.map(|x| x * w * self.params.cmu / (self.params.sigma * self.params.sigma));
            state.cov = &state.cov + dx;
        }

        // Perform step-size sigma update
        let cn = self.params.cs / self.params.damps;
        let sum_square_ps = state.ps.map(|x| x * x).sum();
        let other = cn * (sum_square_ps / self.params.n - 1.) / 2.;
        state.sigma *= f32::min(1.0, other).exp();

        Ok(state)
    }

    /// DONE
    /// Stopping criteria: close (smoothed) to target
    ///
    /// TODO
    /// doctest this
    ///
    fn is_done(&self, state: &CmaesState, step: i32) -> Result<Self::Done> {
        ////////////////
        // TODO
        // Dynamic how many historicals to average
        ////////////////
        let best_y_avg = if state.best_y_hist.len() > 10 {
            let data = state.best_y_hist[state.best_y_hist.len() - 10..].to_vec();
            DVector::from_vec(data).mean()
            // median(data)
        } else {
            let data = state.best_y_hist[..].to_vec();
            DVector::from_vec(data).mean()
            // median(data)
        };

        // println!("Best y fit GLOBAL {:?}", state.best_y_fit.row(0)[0]);
        // println!("Fit Hist (avg) {:?}", &state.best_y_hist);
        // println!();

        ////////////////
        // TODO
        // Dynamic how steps to require
        ////////////////
        if (step > 5) & ((state.best_y_fit.row(0)[0] - best_y_avg).abs() < self.params.tol) {
            println!("\n===> Search stopped due to tolerance change met");
            Ok(true)
        } else {
            Ok(false)
        }
    }
}
