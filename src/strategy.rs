use anyhow::Result;
use nalgebra::min;
use ndarray::{s, Array1, Array2, Axis, Zip};
use ndarray_linalg::{Cholesky, Scalar, UPLO};
use ndarray_rand::{rand_distr::StandardNormal, RandomExt};

use crate::{
    // fitness::Fitness,
    fitness::Fitness,
    params::{CmaesParams, CmaesParamsValid},
    state::CmaesState,
};

#[derive(Debug)]
pub struct Cmaes {
    pub params: CmaesParamsValid,
    // pub state: CmaesState,
}

#[derive(Debug, Clone)]
pub struct Individual {
    pub x: Array1<f32>,
}

#[derive(Debug, Clone)]
pub struct Population {
    pub xs: Array2<f32>,
}

impl Cmaes {
    pub fn new(params: &CmaesParams) -> Result<Cmaes> {
        let params = CmaesParamsValid::validate(params)?;
        Ok(Cmaes { params })
    }

    fn ask_one(&self, params: &CmaesParamsValid, state: &CmaesState) -> Result<Individual> {
        // Generate one individual from params and current state
        // z ~ N(0, I)
        let z: Array1<f32> = Array1::random((params.xstart.len(),), StandardNormal);

        // Rotate towards eigen i.e. y = B @ D_diag^0.5 @ (z*sigma)
        let y: Array1<f32> = state
            .eig_vecs
            .dot(&Array2::from_diag(&state.eig_vals))
            .dot(&z.mapv(|elem| elem * state.sigma));

        // Scale and translate i.e. x =  y + μ
        let x: Array1<f32> = y + &state.mean;

        Ok(Individual { x })
    }

    pub fn ask(&self, state: &mut CmaesState) -> Result<Population> {
        // Prepare before ask population
        state.prepare_ask()?;

        // Create population by looping ask_one
        let popsize = self.params.popsize;
        let mut xs: Array2<f32> =
            Array2::zeros((popsize as usize, self.params.xstart.len() as usize));
        for i in 0..popsize {
            let indiv: Individual = self.ask_one(&self.params, &state)?;
            xs.row_mut(i as usize).assign(&indiv.x);
        }
        Ok(Population { xs })
    }

    pub fn tell(
        &self,
        mut state: CmaesState,
        pop: &mut Population,
        fitness: &mut Fitness,
    ) -> Result<CmaesState> {
        
        // Counts
        state.g += 1;
        state.evals_count += fitness.values.nrows() as i32;

        let xold = state.mean.to_owned();

        // Sort indices of fitness and population by ascending fitness
        let mut indices: Vec<usize> = (0..fitness.values.nrows()).collect();
        indices.sort_by(|&i, &j| {
            fitness.values[[i, 0]]
                .partial_cmp(&fitness.values[[j, 0]])
                .unwrap()
        });

        // Sort population matrix and fitness
        let mut sorted_xs: Array2<f32> = Array2::zeros((pop.xs.nrows(), pop.xs.ncols()));
        let mut sorted_fit: Array2<f32> =
            Array2::zeros((fitness.values.nrows(), fitness.values.ncols()));
        for (new_idx, &original_idx) in indices.iter().enumerate() {
            sorted_xs.row_mut(new_idx).assign(&pop.xs.row(original_idx));
            sorted_fit
                .row_mut(new_idx)
                .assign(&fitness.values.row(original_idx));
        }
        pop.xs.assign(&sorted_xs);
        fitness.values.assign(&sorted_fit);

        // Selection and recombination for evolution
        // Select top μ individuals and their weights
        let y_mu: Array2<f32> = pop.xs.slice(s![..self.params.mu, ..]).t().to_owned();
        let weights_mu: Array2<f32> = self
            .params
            .weights
            .slice(s![..self.params.mu])
            .view()
            .into_shape((self.params.mu as usize, 1))
            .unwrap()
            .to_owned();
        // Compute new weighted mean value
        state.mean = y_mu.dot(&weights_mu).sum_axis(Axis(1));

        // // Update mean of distribution: m = m + cm * σ * y_w

        // Cumulation: update evolution paths
        let y: Array1<f32> = &state.mean - &xold;
        let z: Array1<f32> = state.cov.dot(&y);
        let csn = (self.params.cs * (2. - self.params.cs) * self.params.mueff).sqrt() / state.sigma;

        // Update evolution path sigma
        state.ps = Zip::from(&state.ps)
            .and(&z)
            .map_collect(|&ps_, &z_| (1. - self.params.cs) * ps_ + csn * z_);
        // Update evolution path sigma in-place
        // Zip::from(&mut state.ps)
        // .and(&z)
        // .for_each(|ps_, &x_| {
        //     *ps_ = (1. - self.params.cs) * *ps_ + csn * x_;
        // });

        let ccn = (self.params.cc * (2. - self.params.cc) * self.params.mueff).sqrt() / state.sigma;
        let hsig = state.ps.mapv(|x| x.square()).sum()
            / (state.ps.len() as f32)
            / (1. - (1. - self.params.cs).powi(2 * state.evals_count / self.params.popsize));

        // // Update evolution path covariance
        state.pc = Zip::from(&state.pc)
            .and(&y)
            .map_collect(|&pc_, &y_| (1. - self.params.cs) * pc_ + ccn * hsig * y_);

        let c1a =
            self.params.c1 * (1. - (1. - hsig.square()) * self.params.cc * (2. - self.params.cc));
        state.cov = state
            .cov
            .mapv(|x| x * (1. - c1a - self.params.cmu * self.params.weights.iter().sum::<f32>()));

        // Some helpers for easy broadcast
        let pc_col: Array2<f32> = state.pc.clone().insert_axis(Axis(1));
        let pc_outer: Array2<f32> = pc_col.dot(&pc_col.t()).mapv(|x| x * self.params.c1);
        // Perform the rank-one update
        state.cov = Zip::from(&state.cov)
            .and(&pc_outer)
            .map_collect(|&cov_, &pc_outer_| cov_ + pc_outer_);

        // Adjust covariance matrix
        for (i, w) in self.params.weights.iter().enumerate() {
            let mut w = *w;
            if w < 0. {
                // let mahalanobis_norm_sq = self.params.sigma
                //     / self
                //         .mahalanobis_norm(&mut state, &dx.slice(s![i, ..]).to_owned())
                //         .square();
                w = f32::EPSILON
            }
            let dx: Array1<f32> = &pop.xs.slice(s![i, ..]) - &xold;
            let dx: Array2<f32> = dx.insert_axis(Axis(1));
            let dx: Array2<f32> = dx.dot(&dx.t()).mapv(|x| x * w * self.params.cmu / self.params.sigma.square());
            
            println!("{i}, {:+.4}", &dx);
            println!("{:+.4}", &w);
            
            let update: Array2<f32> = Zip::from(&state.cov)
                .and(&dx)
                .map_collect(|&cov_, &dx_| cov_ + dx_);
            println!("{:+.4}", &update);
            state.cov = state.cov + update;
        }

        // println!("{:+.4}", &state.ps);
        // println!("{:+.4}", &self.params.weights);
        // println!("{:+.4}", &state.pc);
        // println!("{:+.4}", &state.cov);

        // // Adapt step-size sigma
        // let cn = self.params.cs / self.params.damps;
        // let sum_square_ps = state.ps.mapv(|x| x.square()).sum();
        // state.sigma = state.sigma * (f32::min(1., cn * (sum_square_ps / self.params.n - 1.) / 2.));

        Ok(state)
    }

    // fn mahalanobis_norm(&self, state: &mut CmaesState, dx: &Array1<f32>) -> f32 {
    //     // (dx^T * C^-1 * dx)**0.5
    //     dx.dot(&state.cov.mapv(|x| 1. / x).dot(dx)).sqrt()
    // }

    // TODO
    // Reset required variables for next pop
    // pub fn after_tell(...) {
    // }
}
