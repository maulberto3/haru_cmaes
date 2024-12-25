use anyhow::Result;
use ndarray::{s, Array1};

/// Parameters for CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
#[derive(Debug, Clone)]
pub struct CmaesParams {
    pub popsize: i32,         // Population size
    pub xstart: Vec<f32>,     // Initial guess (mean vector)
    pub sigma: f32,           // Step-size (standard deviation)
    pub tol: f32,             // Tolerance for convergence, optional
    pub obj_value: f32,       // Known objective value, optional
    pub zs: f32,              // Enforce zero sparsity for quicker computational results, optional
    pub n: f32,               // Dimension of the problem space (xstart size)
    pub mu: i32,              // Number of parents (best individuals)
    pub weights: Array1<f32>, // Weights for recombination
    pub mueff: f32,           // Effective number of parents
    pub cc: f32,              // Cumulation constant for the rank-one update
    pub cs: f32,              // Cumulation constant for the rank-mu update
    pub c1: f32,              // Learning rate for the rank-one update
    pub cmu: f32,             // Learning rate for the rank-mu update
    pub damps: f32,           // Damping for step-size adaptation
                              // pub lazy_gap_evals: f32, // Gap to postpone eigendecomposition
}

/// Trait defining validation methods for CMA-ES parameters.
pub trait CmaesParamsValidator {
    type Output;
    fn new() -> Result<Self::Output>;
    // Fundamental
    fn set_popsize(self, popsize: i32) -> Result<Self::Output>;
    fn set_xstart(self, xstart: Vec<f32>) -> Result<Self::Output>;
    fn set_sigma(self, sigma: f32) -> Result<Self::Output>;
}

/// Trait for Validated Cmaes Params
impl CmaesParamsValidator for CmaesParams {
    type Output = CmaesParams;

    /// Creates default parameters for the CMA-ES algorithm based on the provided parameters.
    fn new() -> Result<Self::Output> {
        let popsize = 50;
        let xstart = vec![0.0; 50];
        let sigma = 0.75;

        let tol = 0.0001;
        let obj_value = 0.0;

        let zs = 0.05;

        let n = xstart.len() as f32;
        let k = popsize as f32;
        // let chin = n.sqrt() * (1. - 1. / (4. * n) + 1. / (21. * n * n));
        let mu = popsize / 2;

        let _iterable: Vec<f32> = (0..popsize)
            .map(|_x| {
                if _x < mu {
                    (k / 2.0 + 0.5).ln() - ((_x + 1) as f32).ln()
                } else {
                    0.0
                }
            })
            .collect();

        let _weights: Array1<f32> = Array1::from_iter(_iterable);
        let _w_sum = _weights.slice(s![..mu]).sum();
        let weights: Array1<f32> = _weights.mapv(|x| x / _w_sum);

        let mueff: f32 = (weights.slice(s![..mu]).sum() * weights.slice(s![..mu]).sum())
            / weights.slice(s![..mu]).mapv(|x| x * x).sum();

        let cc = (4. + mueff / n) / (n + 4. + 2. * mueff / n);
        let cs = (mueff + 2.) / (n + mueff + 5.);
        let c1 = 2. / ((n + 1.3) * (n + 1.3) + mueff);
        let cmu = (1. - c1).min(2. * (mueff - 2. + 1. / mueff) / ((n + 2.) * (n + 2.) + mueff));
        let damps = 2. * mueff / k + 0.3 + cs;

        // gap to postpone eigendecomposition to achieve O(N**2) per eval
        // 0.5 is chosen such that eig takes 2 times the time of tell in >=20-D
        // let lazy_gap_evals = 0.5 * n * (k) * (c1 + cmu).powi(-1) / (n * n);

        let params = CmaesParams {
            // Fundamental
            popsize,
            xstart,
            sigma,
            // Objective's
            tol,
            obj_value,
            // Computational's
            zs,
            // Others
            n,
            mu,
            weights,
            mueff,
            cc,
            cs,
            c1,
            cmu,
            damps,
        };
        Ok(params)
    }

    fn set_popsize(mut self, popsize: i32) -> Result<Self> {
        self.popsize = popsize;
        Ok(self)
    }

    fn set_xstart(mut self, xstart: Vec<f32>) -> Result<Self::Output> {
        self.xstart = xstart;
        Ok(self)
    }

    fn set_sigma(mut self, sigma: f32) -> Result<Self::Output> {
        self.sigma = sigma;
        Ok(self)
    }
}
