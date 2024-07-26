use core::f32;

use anyhow::Result;

use ndarray::{Array, Array1, Array2};
use ndarray_linalg::Eig;
// use ndarray_rand::RandomExt;
// use rand::distributions::Uniform;

use crate::params::CmaesParams;

/// State for the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) algorithm.
#[derive(Debug, Clone)]
pub struct CmaesState {
    /// Matrix of standard normal random variables.
    pub z: Array2<f32>,
    /// Matrix of candidate solutions.
    pub y: Array2<f32>,
    /// Best candidate solution.
    pub best_y: Array1<f32>,
    /// Fitness values of the best candidates.
    pub best_y_fit: Array1<f32>,
    /// Covariance matrix of the population.
    pub cov: Array2<f32>,
    /// Eigenvectors of the covariance matrix.
    pub eig_vecs: Array2<f32>,
    /// Eigenvalues of the covariance matrix.
    pub eig_vals: Array1<f32>,
    /// Matrix for the inverse square root of the covariance matrix.
    pub inv_sqrt: Array2<f32>,
    /// Mean of the population.
    pub mean: Array1<f32>,
    /// Step-size (standard deviation).
    pub sigma: f32,
    /// Current generation.
    pub g: i32,
    /// Number of evaluations performed.
    pub evals_count: i32,
    /// Evolution path for step-size adaptation.
    pub ps: Array1<f32>,
    /// Evolution path for covariance matrix adaptation.
    pub pc: Array1<f32>,
}

impl CmaesState {
    /// Initializes the state for the CMA-ES algorithm.
    ///
    /// # Parameters
    /// - `params`: The CMA-ES parameters to initialize the state.
    ///
    /// # Returns
    /// A `Result` containing the initialized `CmaesState` or an error if initialization fails.
    pub fn init_state(params: &CmaesParams) -> Result<Self> {
        // Create initial values for the state
        // print!("Creating a new state... ");
        let z: Array2<f32> = Array2::zeros((params.popsize as usize, params.xstart.len()));
        let y: Array2<f32> = Array2::zeros((params.popsize as usize, params.xstart.len()));
        let best_y: Array1<f32> = Array1::zeros(params.xstart.len());
        let best_y_fit: Array1<f32> = Array1::from_elem(1, f32::MAX);
        let cov: Array2<f32> = Array2::eye(params.xstart.len());
        // let cov: Array2<f32> = Array2::random(
        //     (params.xstart.len(), params.xstart.len()),
        //     Uniform::new(0.0, 0.5),
        // );
        let inv_sqrt: Array2<f32> = Array2::eye(params.xstart.len());
        let eig_vecs: Array2<f32> = Array::eye(params.xstart.len());
        let eig_vals: Array1<f32> = Array::from_elem((params.xstart.len(),), 1.0);
        let mean: Array1<f32> = Array1::from_vec(params.xstart.clone());
        let sigma: f32 = params.sigma;
        let g: i32 = 0;
        let evals_count = 0;
        let ps: Array1<f32> = Array1::zeros(params.xstart.len());
        let pc: Array1<f32> = Array1::zeros(params.xstart.len());

        Ok(CmaesState {
            z,
            y,
            best_y,
            best_y_fit,
            cov,
            eig_vecs,
            eig_vals,
            inv_sqrt,
            mean,
            sigma,
            g,
            evals_count,
            ps,
            pc,
        })
    }

    /// Prepares the state by performing eigen decomposition on the covariance matrix.
    ///
    /// # Returns
    /// A `Result` indicating success or failure.
    pub fn prepare_ask(&mut self) -> Result<()> {
        self.eigen_decomposition();
        Ok(())
    }

    /// Performs eigen decomposition on the covariance matrix.
    ///
    /// This method computes the eigenvalues and eigenvectors of the covariance matrix,
    /// adjusts the eigenvalues to ensure numerical stability, and reconstructs the matrix
    /// for use in the CMA-ES algorithm.
    fn eigen_decomposition(&mut self) {
        // Ensure symmetric covariance
        self.cov = (&self.cov + &self.cov.t()) / 2.0;

        // Get eigenvalues and eigenvectors of covariance matrix i.e. C = B * Λ * B^T
        let (eig_vals, eig_vecs) = self.cov.eig().unwrap();

        // Extract real parts of eigenvalues and eigenvectors
        let mut eig_vals: Array1<f32> = eig_vals.mapv(|eig| eig.re);
        let eig_vecs: Array2<f32> = eig_vecs.mapv(|vec| vec.re);

        // Ensure positive numbers
        eig_vals.map_inplace(|elem| {
            if *elem < 0.0 {
                *elem = 0.1 // TODO: instability here with f32::EPSILON, try other values
            } else if *elem > 10. {
                *elem = 10.; // Clamp to a maximum value to avoid overflow
            }
            //  else { *elem = *elem }
        });

        // Short-hand for inverse square root of C
        self.inv_sqrt = Array2::from_diag(&eig_vals.mapv(|elem| elem.powf(-0.5)));
        self.inv_sqrt = eig_vecs.dot(&self.inv_sqrt).dot(&eig_vecs.t());

        self.eig_vecs = eig_vecs;
        self.eig_vals = eig_vals;
    }
}
