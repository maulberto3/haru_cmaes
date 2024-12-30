use anyhow::Result;
use nalgebra::{DMatrix, DVector, SymmetricEigen};
// use ndarray_rand::RandomExt;
// use rand::distributions::Uniform;
use crate::params::CmaesParams;
// use crate::utils::into_f_major;

/// State for the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) algorithm.
#[derive(Debug, Clone)]
pub struct CmaesState {
    pub z: DMatrix<f32>,          // Matrix of standard normal random variables.
    pub y: DMatrix<f32>,          // Matrix of candidate solutions.
    pub best_y: DVector<f32>,     // Best candidate solution.
    pub best_y_fit: DVector<f32>, // Fitness values of the best candidates.
    pub cov: DMatrix<f32>,        // Covariance matrix of the population.
    pub eig_vecs: DMatrix<f32>,   // Eigenvectors of the covariance matrix.
    pub eig_vals: DVector<f32>,   // Eigenvalues of the covariance matrix.
    pub inv_sqrt: DMatrix<f32>,   // Matrix for the inverse square root of the covariance matrix.
    pub mean: DVector<f32>,       // Mean of the population.
    pub sigma: f32,               // Step-size (standard deviation).
    pub g: i32,                   // Curren generation.
    pub evals_count: i32,         // Number of evaluations performed.
    pub ps: DVector<f32>,         // Evolution path for step-size adaptation.
    pub pc: DVector<f32>,         // Evolution path for covariance matrix adaptation.
}

/// Trait for Cmaes State
pub trait CmaesStateLogic {
    type NewState;

    fn init_state(params: &CmaesParams) -> Result<Self::NewState>;
    fn prepare_ask(&mut self, params: &CmaesParams) -> Result<()>;
    fn eigen_decomposition(&mut self, params: &CmaesParams) -> Result<()>;
}

impl CmaesStateLogic for CmaesState {
    /// Initializes the state for the CMA-ES algorithm.
    type NewState = CmaesState;

    fn init_state(params: &CmaesParams) -> Result<Self::NewState> {
        // Create initial values for the state
        let z: DMatrix<f32> = DMatrix::zeros(params.popsize as usize, params.xstart.len());
        let y: DMatrix<f32> = DMatrix::zeros(params.popsize as usize, params.xstart.len());
        let best_y: DVector<f32> = DVector::zeros(params.xstart.len());
        let best_y_fit: DVector<f32> = DVector::from_element(1, f32::MAX);
        let cov: DMatrix<f32> = DMatrix::identity(params.xstart.len(), params.xstart.len());
        let inv_sqrt: DMatrix<f32> = DMatrix::identity(params.xstart.len(), params.xstart.len());
        let eig_vecs: DMatrix<f32> = DMatrix::identity(params.xstart.len(), params.xstart.len());
        let eig_vals: DVector<f32> = DVector::identity(params.xstart.len());
        let mean: DVector<f32> = DVector::from_vec(params.xstart.clone());
        let sigma: f32 = params.sigma;
        let g: i32 = 0;
        let evals_count = 0;
        let ps: DVector<f32> = DVector::zeros(params.xstart.len());
        let pc: DVector<f32> = DVector::zeros(params.xstart.len());

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
    fn prepare_ask(&mut self, params: &CmaesParams) -> Result<()> {
        let _ = self.eigen_decomposition(params);
        Ok(())
    }

    /// Performs eigen decomposition on the covariance matrix.
    ///
    /// This method computes the eigenvalues and eigenvectors of the covariance matrix,
    /// adjusts the eigenvalues to ensure numerical stability, and reconstructs the matrix
    /// for use in the CMA-ES algorithm.
    fn eigen_decomposition(&mut self, params: &CmaesParams) -> Result<()> {
        // Ensure symmetric covariance
        // self.cov.zip_mut_with(&self.cov.t().to_owned(), |x, &t| {
        //     *x = (*x + t) / 2.0;
        // });
        self.cov = (&self.cov + &self.cov.transpose()) / 2.0;

        // // Enforce sparsity for matrix eigen efficiency
        // self.cov.map_inplace(|x| {
        //     if x.abs() < params.zs {
        //         *x = 0.0
        //     }
        // });
        self.cov.iter_mut().for_each(|x| {
            if x.abs() < params.zs {
                *x = 0.0;
            }
        });

        ////////////////
        // TODO
        // Allow flag for verbose state
        // maybe with tracing
        ////////////////
        // println!("{:?}", &self);
        // dbg!(&self);
        // println!("");
        ////////////////

        // Perform eigen decomposition: C = B * Λ * B^T
        #[cfg(any(
            feature = "openblas",
            feature = "netlib",
            feature = "accelerate",
            feature = "intel-mkl"
        ))]
        let mut eigen =
            nalgebra_lapack::SymmetricEigen::try_new(self.cov.clone()).ok_or(PosDefCovError)?;

        #[cfg(not(any(
            feature = "openblas",
            feature = "netlib",
            feature = "accelerate",
            feature = "intel-mkl"
        )))]
        let eigen = SymmetricEigen::try_new(self.cov.clone(), 1e-20, 0).unwrap();

        let mut eig_vals: DVector<f32> = eigen.eigenvalues;
        let eig_vecs: DMatrix<f32> = eigen.eigenvectors;

        // Ensure positive eigenvalues
        eig_vals.iter_mut().for_each(|eig| {
            if *eig < 0.0 {
                *eig = 0.1; // Adjust negative eigenvalues
            } else if *eig > 10.0 {
                *eig = 10.0; // Clamp to a maximum value
            }
        });

        // Calculate the inverse square root of eigenvalues
        let inv_sqrt_diag = DMatrix::from_diagonal(&eig_vals.map(|eig| eig.powf(-0.5)));
        self.inv_sqrt = &eig_vecs * &inv_sqrt_diag * eig_vecs.transpose();

        // Store eigenvalues and eigenvectors
        self.eig_vecs = eig_vecs;
        self.eig_vals = eig_vals;

        Ok(())
    }
}
