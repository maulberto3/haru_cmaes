use anyhow::Result;

use ndarray::{Array, Array1, Array2};
use ndarray_linalg::{eig, Eig};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

use crate::params::CmaesParams;

#[derive(Debug, Clone)]
pub struct CmaesState {
    pub z: Array2<f32>,
    pub y: Array2<f32>,
    pub cov: Array2<f32>,
    pub eig_vecs: Array2<f32>,
    pub eig_vals: Array1<f32>,
    pub inv_sqrt: Array2<f32>,
    pub mean: Array1<f32>,
    pub sigma: f32,
    pub g: i32,
    pub evals_count: i32,
    pub ps: Array1<f32>,
    pub pc: Array1<f32>,
    // pub best: Vec<
}

impl CmaesState {
    pub fn init_state(params: &CmaesParams) -> Result<Self> {
        // Create initial values for the state
        print!("Creating a new state... ");
        let z: Array2<f32> = Array2::zeros((params.popsize as usize, params.xstart.len()));
        let y: Array2<f32> = Array2::zeros((params.popsize as usize, params.xstart.len()));
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
        println!("Done.");

        Ok(CmaesState {
            z,
            y,
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

    pub fn prepare_ask(&mut self) -> Result<()> {
        self.eigen_decomposition();
        Ok(())
    }

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

        // Reconstruct the covariance matrix: C = B * D^-.5 * B^T
        self.inv_sqrt = Array2::from_diag(&eig_vals.mapv(|elem| elem.powf(-0.5)));
        self.inv_sqrt = eig_vecs.dot(&self.inv_sqrt).dot(&eig_vecs.t());

        self.eig_vecs = eig_vecs;
        self.eig_vals = eig_vals;
    }
}
