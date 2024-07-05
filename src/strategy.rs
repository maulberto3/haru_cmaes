use anyhow::Result;

use ndarray::{s, Array1, Array2, ArrayView1, Axis};
use ndarray_linalg::Norm;
use ndarray_rand::{rand_distr::StandardNormal, RandomExt};

use crate::{
    fitness::Fitness,
    params::{CmaesParams, CmaesParamsValid},
    state::CmaesState,
};
// use crate::state::CmaesState;

#[derive(Debug)]
pub struct Cmaes {
    pub params_valid: CmaesParamsValid,
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
    pub fn new(params: &CmaesParams) -> Result<Self> {
        // Instantiate Cmaes
        let params_valid = CmaesParamsValid::validate(params)?;
        Ok(Cmaes { params_valid })
    }

    fn ask_one(&self, params: &CmaesParamsValid, state: &CmaesState) -> Result<Individual> {
        // Generate one individual from params and current state
        // z ~ N(0, I)
        let z: Array1<f32> = Array1::random((params.mean.len(),), StandardNormal);

        // Rotate towards eigen i.e. y = B * D_diag * z
        let y: Array1<f32> = state
            .eig_vecs
            .dot(&Array2::from_diag(&state.eig_vals))
            .dot(&z);

        // Scale and translate i.e. x =  σ * y + μ
        let x: Array1<f32> = y.mapv(|elem| elem * state.sigma) + &state.mean;

        Ok(Individual { x })
    }

    pub fn ask(&self, state: &mut CmaesState) -> Result<Population> {
        // Prepare before ask population
        state.prepare_ask()?;

        // Create population by looping ask_one
        let popsize = self.params_valid.popsize;
        let mut xs: Array2<f32> =
            Array2::zeros((popsize as usize, self.params_valid.mean.len() as usize));
        for i in 0..popsize {
            let indiv: Individual = self.ask_one(&self.params_valid, &state)?;
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
        // Increment step count
        state.g += 1;

        // Sort indices of fitness and population by ascending fitness
        let mut indices: Vec<usize> = (0..fitness.fit.len()).collect();
        indices.sort_by(|&i, &j| fitness.fit[i].partial_cmp(&fitness.fit[j]).unwrap());

        // Sort population matrix and fitness
        let mut sorted_xs: Array2<f32> = Array2::zeros((pop.xs.nrows(), pop.xs.ncols()));
        let mut sorted_fit: Array1<f32> = Array1::zeros(fitness.fit.len());
        for (new_idx, &original_idx) in indices.iter().enumerate() {
            sorted_xs.row_mut(new_idx).assign(&pop.xs.row(original_idx));
            sorted_fit[new_idx] = fitness.fit[original_idx];
        }
        pop.xs.assign(&sorted_xs);
        fitness.fit.assign(&sorted_fit);

        // println!("");
        // dbg!(&pop.xs);
        // dbg!(&fitness.fit);
        // println!("");

        // Normalize population: y = (x - m) / σ
        pop.xs.axis_iter_mut(Axis(0)).for_each(|mut row| {
            row -= &state.mean;
            row /= state.sigma;
        });

        // Selection and recombination for evolution
        // Select top μ individuals and their weights
        let y_mu: Array2<f32> = pop.xs.slice(s![..self.params_valid.mu, ..]).t().to_owned();
        let weights_mu: Array2<f32> = self
            .params_valid
            .weights_prime
            .slice(s![..self.params_valid.mu])
            .view()
            .into_shape((self.params_valid.mu, 1))
            .unwrap()
            .to_owned();
        let y_w: Array1<f32> = y_mu.dot(&weights_mu).sum_axis(Axis(1));
        // Update mean of distribution: m = m + cm * σ * y_w
        state.mean = state.mean + y_w.mapv(|x| x * self.params_valid.cm * self.params_valid.sigma);

        // Step-size control
        // Compute the inverse square root of the covariance matrix C (using its eigendecomposition i.e. C^(-1/2) = B * D^(-1) * B^T)
        let c_2: Array2<f32> = state
            .eig_vecs
            .dot(&Array2::from_diag(&state.eig_vals.mapv(|eigv| 1.0 / eigv)))
            .dot(&state.eig_vecs.t());

        // Update the evolution path for for the covariance matrix (using the inverse square root of C and the weighted sum of individuals (y_w))
        state.p_sigma = (1.0 - self.params_valid.c_sigma) * state.p_sigma
            + (self.params_valid.c_sigma
                * (2.0 - self.params_valid.c_sigma)
                * self.params_valid.mu_eff)
                .sqrt()
                * c_2.dot(&y_w);

        // Compute the norm of the evolution path for sigma (p_sigma)
        let norm_p_sigma: f32 = (state.p_sigma).norm_l2();

        // Update the global step-size control parameter (sigma)
        state.sigma = state.sigma
            * ((self.params_valid.c_sigma / self.params_valid.d_sigma)
                * (norm_p_sigma / self.params_valid.chi_n - 1.0))
                .exp();

        // Covariance matrix adaption
        // Calculate the left condition for h_sigma
        let h_sigma_cond_left: f32 =
            norm_p_sigma / (1.0 - (1.0 - self.params_valid.c_sigma).powi(2 * (state.g + 1))).sqrt();
        // Calculate the right condition for h_sigma
        let h_sigma_cond_right: f32 =
            (1.4 + 2.0 / (self.params_valid.num_dims + 1.0)) * self.params_valid.chi_n;
        // Determine h_sigma (based on comparing the left and right conditions)
        let h_sigma: f32 = match h_sigma_cond_left < h_sigma_cond_right {
            true => 1.0,
            false => 0.0,
        };

        // (eq.45)
        // Update evolution path of covariance matrix adaptation
        state.p_c = (1.0 - self.params_valid.cc) * &state.p_c
            + h_sigma
                * (self.params_valid.cc * (2.0 - self.params_valid.cc) * self.params_valid.mu_eff)
                    .sqrt()
                * &y_w;

        // (eq.46)
        let w_io = &self.params_valid.weights
            * &self
                .params_valid
                .weights
                .mapv(|w| if w >= 0.0 { 1.0 } else { 0.0 });

        let delta_h_sigma = (1.0 - h_sigma) * self.params_valid.cc * (2.0 - self.params_valid.cc);

        // (eq.47)
        let rank_one_col = state.p_c.view().into_shape((state.p_c.len(), 1)).unwrap();
        let rank_one_row = state.p_c.view().into_shape((1, state.p_c.len())).unwrap();
        let rank_one = rank_one_col.dot(&rank_one_row);

        let mut rank_mu: Array2<f32> = Array2::zeros((pop.xs.ncols(), pop.xs.ncols()));

        // Iterate over weights and population vectors
        for (w, y) in w_io.iter().zip(pop.xs.axis_iter(Axis(0))) {
            let outer_col = y.view().into_shape((y.len(), 1)).unwrap();
            let outer_row = y.view().into_shape((1, y.len())).unwrap();
            let outer = outer_col.dot(&outer_row);
            rank_mu = rank_mu + *w * outer;
        }

        //
        state.cov = (1.0 + self.params_valid.c1 * delta_h_sigma
            - self.params_valid.c1
            - self.params_valid.cmu * self.params_valid.weights.sum())
            * state.cov
            + rank_one.mapv(|x| x * self.params_valid.c1)
            + rank_mu.mapv(|x| x * self.params_valid.cmu);

        // Learning rate adaptation (enhancement)

        Ok(state)
    }

    // TODO
    // Reset required variables for next pop
    // pub fn after_tell(...) {
    // }

    // TODO
    // Whether the draw individual is within bounds supplied
    // If not, re-draw given self._n_max_resampling, or
    // ultimately, just clip values within bounds
    // fn _is_feasible()
    // fn _repair_infeasible_params

    // TODO
    // As per repo example:
    // ```
    // def main():
    //     optimizer = CMA(mean=np.zeros(2), sigma=1.3)
    //     print(" g    f(x1,x2)     x1      x2  ")
    //     print("===  ==========  ======  ======")
    //     while True:
    //         solutions = []
    //         for _ in range(optimizer.population_size):
    //             x = optimizer.ask()
    //             value = quadratic(x[0], x[1])
    //             solutions.append((x, value))
    //             print(
    //                 f"{optimizer.generation:3d}  {value:10.5f}"
    //                 f"  {x[0]:6.2f}  {x[1]:6.2f}"
    //             )
    //         optimizer.tell(solutions)
    //         if optimizer.should_stop():
    //             break
    // ```
    // TODO: make one go for population, no loop
    // as suggested in repo example, attached above
    // If ask_one is independent, try to paralellize
}
