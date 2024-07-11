use anyhow::Result;
use ndarray::{s, Array1, Array2, Axis, Zip};
use ndarray_linalg::Scalar;
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

        // Update mean of distribution: m = m + cm * σ * y_w

        // Cumulation: update evolution paths
        let y: Array1<f32> = &state.mean - xold;
        let z: Array1<f32> = state.cov.mapv(|x| x.powf(-0.5)).dot(&y);
        let csn = (self.params.cs * (2. - self.params.cs) * self.params.mueff).sqrt() / state.sigma;

        // Update evolution path sigma
        state.ps = Zip::from(&state.ps)
            .and(&z)
            .map_collect(|&ps_, &x_| (1. - self.params.cs) * ps_ + csn * x_);

        let ccn = (self.params.cc * (2. - self.params.cc) * self.params.mueff).sqrt() / state.sigma;
        let hsig = state.ps.mapv(|x| x.square()).sum()
            / (state.ps.len() as f32)
            / (1. - (1. - self.params.cs).powi(2 * state.evals_count / self.params.popsize));

        // Update evolution path covariance
        state.pc = Zip::from(&state.pc)
            .and(&y)
            .map_collect(|&pc_, &y_| (1. - self.params.cs) * pc_ + ccn * hsig * y_);

        // Adjust covariance matrix
        let c1a =
            self.params.c1 * (1. - (1. - hsig.square())) * self.params.cc * (2. - self.params.cc);
        state.cov = state.cov.mapv(|x| {
            1. - c1a - self.params.cmu * self.params.weights.iter().sum::<f32>()
        });
//         let outer_product = b.to_owned().into_shape((b.len(), 1)).unwrap() // Convert to column vector
//         .dot(&b.to_owned().into_shape((1, b.len())).unwrap()); // Convert to row vector

// c += &outer_product.mapv(|elem| factor * elem);






        // state.mean = state.mean + y_w.mapv(|x| x * self.params.cm * self.params.sigma);

        // // println!("");
        // // dbg!(&pop.xs);
        // // dbg!(&fitness.values);
        // // println!("");

        // // Normalize population: y = (x - m) / σ
        // pop.xs.axis_iter_mut(Axis(0)).for_each(|mut row| {
        //     row -= &state.mean;
        //     row /= state.sigma;
        // });

        // // Step-size control
        // // Compute the inverse square root of the covariance matrix C (using its eigendecomposition i.e. C^(-1/2) = B * D^(-1) * B^T)
        // let c_2: Array2<f32> = state
        //     .eig_vecs
        //     .dot(&Array2::from_diag(&state.eig_vals.mapv(|eigv| 1.0 / eigv)))
        //     .dot(&state.eig_vecs.t());

        // // Update the evolution path for for the covariance matrix (using the inverse square root of C and the weighted sum of individuals (y_w))
        // state.p_sigma = (1.0 - self.params.c_sigma) * state.p_sigma
        //     + (self.params.c_sigma
        //         * (2.0 - self.params.c_sigma)
        //         * self.params.mu_eff)
        //         .sqrt()
        //         * c_2.dot(&y_w);

        // // Compute the norm of the evolution path for sigma (p_sigma)
        // let norm_p_sigma: f32 = (state.p_sigma).norm_l2();

        // // Update the global step-size control parameter (sigma)
        // state.sigma = state.sigma
        //     * ((self.params.c_sigma / self.params.d_sigma)
        //         * (norm_p_sigma / self.params.chi_n - 1.0))
        //         .exp();

        // // Covariance matrix adaption
        // // Calculate the left condition for h_sigma
        // let h_sigma_cond_left: f32 =
        //     norm_p_sigma / (1.0 - (1.0 - self.params.c_sigma).powi(2 * (state.g + 1))).sqrt();
        // // Calculate the right condition for h_sigma
        // let h_sigma_cond_right: f32 =
        //     (1.4 + 2.0 / (self.params.num_dims + 1.0)) * self.params.chi_n;
        // // Determine h_sigma (based on comparing the left and right conditions)
        // let h_sigma: f32 = match h_sigma_cond_left < h_sigma_cond_right {
        //     true => 1.0,
        //     false => 0.0,
        // };

        // // (eq.45)
        // // Update evolution path of covariance matrix adaptation
        // state.p_c = (1.0 - self.params.cc) * &state.p_c
        //     + h_sigma
        //         * (self.params.cc * (2.0 - self.params.cc) * self.params.mu_eff)
        //             .sqrt()
        //         * &y_w;

        // // (eq.46)
        // let w_io = &self.params.weights
        //     * &self
        //         .params_valid
        //         .weights
        //         .mapv(|w| if w >= 0.0 { 1.0 } else { 0.0 });

        // let delta_h_sigma = (1.0 - h_sigma) * self.params.cc * (2.0 - self.params.cc);

        // // (eq.47)
        // let rank_one_col = state.p_c.view().into_shape((state.p_c.len(), 1)).unwrap();
        // let rank_one_row = state.p_c.view().into_shape((1, state.p_c.len())).unwrap();
        // let rank_one = rank_one_col.dot(&rank_one_row);

        // let mut rank_mu: Array2<f32> = Array2::zeros((pop.xs.ncols(), pop.xs.ncols()));

        // // Iterate over weights and population vectors
        // for (w, y) in w_io.iter().zip(pop.xs.axis_iter(Axis(0))) {
        //     let outer_col = y.view().into_shape((y.len(), 1)).unwrap();
        //     let outer_row = y.view().into_shape((1, y.len())).unwrap();
        //     let outer = outer_col.dot(&outer_row);
        //     rank_mu = rank_mu + *w * outer;
        // }

        // //
        // state.cov = (1.0 + self.params.c1 * delta_h_sigma
        //     - self.params.c1
        //     - self.params.cmu * self.params.weights.sum())
        //     * state.cov
        //     + rank_one.mapv(|x| x * self.params.c1)
        //     + rank_mu.mapv(|x| x * self.params.cmu);

        // // Learning rate adaptation (enhancement)

        Ok(state)
    }

    // TODO
    // Reset required variables for next pop
    // pub fn after_tell(...) {
    // }
}
