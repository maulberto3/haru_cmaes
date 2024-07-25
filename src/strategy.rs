use anyhow::Result;
use ndarray::{s, Array1, Array2, Axis, Zip};
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
pub struct PopulationZ {
    pub z: Array2<f32>,
}

#[derive(Debug, Clone)]
pub struct PopulationY {
    pub y: Array2<f32>,
}

impl Cmaes {
    pub fn new(params: &CmaesParams) -> Result<Cmaes> {
        let params = CmaesParamsValid::validate(params)?;
        Ok(Cmaes { params })
    }

    fn ask_z(&self, params: &CmaesParamsValid, state: &mut CmaesState) -> Result<PopulationZ> {
        // Generate a matrix of standard normal random variables
        let z: Array2<f32> = Array2::random(
            (params.popsize as usize, params.xstart.len()),
            StandardNormal,
        );
        state.z = z.to_owned();
        Ok(PopulationZ { z })
    }

    pub fn ask(&self, state: &mut CmaesState) -> Result<PopulationY> {
        // Prepare before ask population
        state.prepare_ask()?;

        // Ask z
        let z: Array2<f32> = self.ask_z(&self.params, state)?.z;

        // Transform z to y
        let eig_vals_sqrt: Array2<f32> = Array2::from_diag(&state.eig_vals.mapv(f32::sqrt));
        let eigenvectors: Array2<f32> = state.eig_vecs.to_owned();

        let scaled_z: Array2<f32> = (z * state.sigma).dot(&eig_vals_sqrt);
        let rotated_z: Array2<f32> = scaled_z.dot(&eigenvectors.t());

        // Add the mean vector xmean to each row of scaled_y
        let y: Array2<f32> = &rotated_z + &state.mean.broadcast(rotated_z.raw_dim()).unwrap();
        state.y = y.to_owned();

        Ok(PopulationY { y })
    }

    pub fn tell(
        &self,
        mut state: CmaesState,
        pop: &mut PopulationY,
        fitness: &mut Fitness,
    ) -> Result<CmaesState> {
        // Book-keeping
        state.g += 1;
        state.evals_count += fitness.values.nrows() as i32;
        let xold = state.mean.to_owned();

        // Sort by fitness...
        let mut indices: Vec<usize> = (0..fitness.values.nrows()).collect();
        indices.sort_by(|&i, &j| {
            fitness.values[[i, 0]]
                .partial_cmp(&fitness.values[[j, 0]])
                .unwrap()
        });
        // ...both fitness and population
        let mut sorted_xs: Array2<f32> = Array2::zeros((pop.y.nrows(), pop.y.ncols()));
        let mut sorted_fit: Array2<f32> =
            Array2::zeros((fitness.values.nrows(), fitness.values.ncols()));
        for (new_idx, &original_idx) in indices.iter().enumerate() {
            sorted_xs.row_mut(new_idx).assign(&pop.y.row(original_idx));
            sorted_fit
                .row_mut(new_idx)
                .assign(&fitness.values.row(original_idx));
        }
        pop.y.assign(&sorted_xs);
        fitness.values.assign(&sorted_fit);

        // Best current y
        if &fitness.values[[0, 0]] < &state.best_y_fit[0] {
            state.best_y = pop.y.slice(s![0, ..]).to_owned();
            state.best_y_fit = fitness.values.slice(s![0, ..]).to_owned();
        }

        // Selection and recombination for evolution
        // Select top μ individuals and their weights
        let y_mu: Array2<f32> = pop.y.slice(s![..self.params.mu, ..]).to_owned();
        let weights_mu: Array2<f32> = self
            .params
            .weights
            .slice(s![..self.params.mu])
            .view()
            .into_shape((1, self.params.mu as usize))
            .unwrap()
            .to_owned();

        // Compute new weighted mean value
        let y_w: Array2<f32> = y_mu.t().dot(&weights_mu.t());
        let y_w: Array1<f32> = y_w.into_shape(y_mu.ncols()).unwrap();
        state.mean = y_w;

        // Update evolution path sigma
        let new_y: Array1<f32> = &state.mean - &xold;
        let new_z: Array1<f32> = state.inv_sqrt.dot(&new_y);
        let csn = (self.params.cs * (2. - self.params.cs) * self.params.mueff).sqrt() / state.sigma;
        state.ps = Zip::from(&state.ps)
            .and(&new_z)
            .map_collect(|&ps_, &z_| (1. - self.params.cs) * ps_ + csn * z_);

        // Update evolution path sigma in-place
        // Zip::from(&mut state.ps)
        // .and(&z)
        // .for_each(|ps_, &x_| {
        //     *ps_ = (1. - self.params.cs) * *ps_ + csn * x_;
        // });

        // Update evolution path covariance
        let ccn = (self.params.cc * (2. - self.params.cc) * self.params.mueff).sqrt() / state.sigma;
        let hsig = state.ps.mapv(|x| x * x).sum()
            / (state.ps.len() as f32)
            / (1. - (1. - self.params.cs).powi(2 * state.evals_count / self.params.popsize));
        state.pc = Zip::from(&state.pc)
            .and(&new_y)
            .map_collect(|&pc_, &y_| (1. - self.params.cs) * pc_ + ccn * hsig * y_);

        // Adapt covariance matrix C
        let c1a =
            self.params.c1 * (1. - (1. - hsig * hsig) * self.params.cc * (2. - self.params.cc));
        state.cov = state.cov.mapv(|x| x * (1. - c1a - self.params.cmu)); // * self.params.weights.iter().sum::<f32>() == 1.0...
                                                                          // // Some helpers for easy broadcast
        let pc_outer: Array2<f32> = state.pc.clone().insert_axis(Axis(1));
        let pc_outer: Array2<f32> = pc_outer.dot(&pc_outer.t()).mapv(|x| x * self.params.c1);
        // Perform the rank-one update
        state.cov = Zip::from(&state.cov)
            .and(&pc_outer)
            .map_collect(|&cov_, &pc_outer_| cov_ + pc_outer_);

        // Perform the rank-mu update
        for (i, w) in self.params.weights.iter().enumerate() {
            let mut w = *w;
            if w < 0. {
                w = 0.001
            }
            let dx: Array1<f32> = &pop.y.slice(s![i, ..]) - &xold;
            let dx: Array2<f32> = dx.insert_axis(Axis(1));
            let dx: Array2<f32> = dx.dot(&dx.t());
            let dx: Array2<f32> =
                dx.mapv(|x| x * w * self.params.cmu / (self.params.sigma * self.params.sigma));
            state.cov = &state.cov + dx;
        }

        // Adapt step-size sigma
        let cn = self.params.cs / self.params.damps;
        let sum_square_ps = state.ps.mapv(|x| x * x).sum();
        let other = cn * (sum_square_ps / self.params.n - 1.) / 2.;
        state.sigma = state.sigma * f32::min(1.0, other).exp();

        Ok(state)
    }
}
