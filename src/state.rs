// use anyhow::Result;

// use ndarray::{Array, Array1, Array2};
// use ndarray_linalg::Eig;

// use crate::params::CmaesParams;

// #[derive(Debug, Clone)]
// pub struct CmaesState {
//     pub cov: Array2<f32>,
//     pub eig_vecs: Array2<f32>,
//     pub eig_vals: Array1<f32>,
//     pub mean: Array1<f32>,
//     pub sigma: f32,
//     pub g: i32,
//     pub p_sigma: Array1<f32>,
//     pub p_c: Array1<f32>,
//     // pub rng: StdRng,
// }

// impl CmaesState {
//     pub fn init_state(params: &CmaesParams) -> Result<Self> {
//         // Create initial values for the state
//         print!("Creating a new state... ");
//         let cov: Array2<f32> = Array2::eye(params.mean.len());
//         // let cov: Array2<f32> = Array2::random(
//         //     (params.mean.len(), params.mean.len()),
//         //     Uniform::new(-1.0, 1.0),
//         // );
//         let eig_vecs: Array2<f32> = Array::eye(params.mean.len());
//         let eig_vals: Array1<f32> = Array::from_elem((params.mean.len(),), 1.0);
//         let mean: Array1<f32> = Array1::from_vec(params.mean.clone());
//         let sigma: f32 = params.sigma;
//         let g: i32 = 0;
//         let p_sigma: Array1<f32> = Array1::zeros(params.mean.len());
//         let p_c: Array1<f32> = Array1::zeros(params.mean.len());
//         println!("Done.");

//         Ok(CmaesState {
//             cov,
//             eig_vecs,
//             eig_vals,
//             mean,
//             sigma,
//             g,
//             p_sigma,
//             p_c,
//         })
//     }

//     pub fn prepare_ask(&mut self) -> Result<()> {
//         // TODO: this only once before first ask
//         // original code probes when B and D are None
//         // They are set to None sometime during tell method
//         self.eigen_decomposition();
//         Ok(())
//     }

//     fn eigen_decomposition(&mut self) {
//         // Ensure symmetric covariance
//         self.cov = (&self.cov + &self.cov.t()) / 2.0;

//         // Get eigenvalues and eigenvectors of covariance matrix i.e. C = B * Λ * B^T
//         let (eig_vals_2, eig_vecs) = self.cov.eig().unwrap();

//         // Extract real parts of eigenvalues and eigenvectors
//         let eig_vals_2: Array1<f32> = eig_vals_2.mapv(|eig| eig.re);
//         let eig_vecs: Array2<f32> = eig_vecs.mapv(|vec| vec.re);

//         // Convert to positive numbers (negative magnitudes dropped): TODO: why?
//         // And take sqrt of them i.e. D = sqrt(max(Λ, 0))
//         let mut eig_vals = eig_vals_2.clone();
//         eig_vals.map_inplace(|elem| {
//             if *elem < 0.0 {
//                 *elem = (f32::EPSILON).sqrt()
//             } else {
//                 *elem = (*elem).sqrt()
//             }
//         });

//         // Reconstruct the covariance matrix: C = B * diag(Λ) * B^T
//         self.cov = eig_vecs
//             .dot(&Array2::from_diag(&eig_vals.mapv(|elem| elem.powi(2))))
//             .dot(&eig_vecs.t());

//         self.eig_vecs = eig_vecs;
//         self.eig_vals = eig_vals;
//     }
// }
