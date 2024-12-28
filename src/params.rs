use anyhow::Result;
use ndarray::{s, Array1};

/// Parameters for CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
#[derive(Debug, Clone)]
pub struct CmaesParams {
    pub popsize: i32,         // Population size
    pub xstart: Vec<f32>,     // Initial guess (mean vector)
    pub sigma: f32,           // Step-size (standard deviation)
    pub tol: f32,             // Tolerance for convergence, optional
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
    // Helper
    fn update_dependent_params(&mut self);
}

/// Trait for Validated Cmaes Params
impl CmaesParamsValidator for CmaesParams {
    type Output = CmaesParams;

    /// Creates default parameters for the CMA-ES algorithm based on the provided parameters.
    ///
    /// Example
    ///
    /// ```rust
    /// use haru_cmaes::{CmaesParams, CmaesParamsValidator};
    ///
    /// let params = CmaesParams::new();
    ///
    /// assert!(params.is_ok());
    /// ```
    fn new() -> Result<Self::Output> {
        let popsize = 10;
        let xstart = vec![0.0; 6];
        let sigma = 0.75;
        let tol = 0.01;
        let zs = 0.01;
        // Update these after setters
        let n = xstart.len() as f32;
        let mu = popsize / 2;
        let k = popsize as f32;
        let iterable: Vec<f32> = (0..popsize)
            .map(|x| {
                if x < mu {
                    (k / 2.0 + 0.5).ln() - ((x + 1) as f32).ln()
                } else {
                    0.0
                }
            })
            .collect();
        let weights: Array1<f32> = Array1::from_iter(iterable);
        let w_sum = weights.slice(s![..mu]).sum();
        let weights: Array1<f32> = weights.mapv(|x| x / w_sum);
        let mueff: f32 = (weights.slice(s![..mu]).sum() * weights.slice(s![..mu]).sum())
            / weights.slice(s![..mu]).mapv(|x| x * x).sum();
        let cc = (4. + mueff / n) / (n + 4. + 2. * mueff / n);
        let cs = (mueff + 2.) / (n + mueff + 5.);
        let c1 = 2. / ((n + 1.3) * (n + 1.3) + mueff);
        let cmu = (1. - c1).min(2. * (mueff - 2. + 1. / mueff) / ((n + 2.) * (n + 2.) + mueff));
        let damps = 2. * mueff / k + 0.3 + cs;

        let params = CmaesParams {
            // Fundamental
            popsize,
            xstart,
            sigma,
            // Objective's
            tol,
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

    /// Updates all parameters that depend on other fields.
    fn update_dependent_params(&mut self) {
        self.n = self.xstart.len() as f32;
        self.mu = self.popsize / 2;
        let k = self.popsize as f32;
        let iterable: Vec<f32> = (0..self.popsize)
            .map(|x| {
                if x < self.mu {
                    (k / 2.0 + 0.5).ln() - ((x + 1) as f32).ln()
                } else {
                    0.0
                }
            })
            .collect();
        let weights: Array1<f32> = Array1::from_iter(iterable);
        let w_sum = weights.slice(s![..self.mu]).sum();
        self.weights = weights.mapv(|x| x / w_sum);
        self.mueff = (self.weights.slice(s![..self.mu]).sum().powi(2))
            / self.weights.slice(s![..self.mu]).mapv(|x| x * x).sum();
        self.cc = (4. + self.mueff / self.n) / (self.n + 4. + 2. * self.mueff / self.n);
        self.cs = (self.mueff + 2.) / (self.n + self.mueff + 5.);
        self.c1 = 2. / ((self.n + 1.3).powi(2) + self.mueff);
        self.cmu = (1. - self.c1)
            .min(2. * (self.mueff - 2. + 1. / self.mueff) / ((self.n + 2.).powi(2) + self.mueff));
        self.damps = 2. * self.mueff / k + 0.3 + self.cs;
    }

    /// Sets population size
    ///
    /// Example
    ///
    /// ```rust
    /// use haru_cmaes::{CmaesParams, CmaesParamsValidator};
    ///
    /// let params = CmaesParams::new()
    ///     .and_then(|p| p.set_popsize(15));
    ///
    /// assert!(params.is_ok());
    /// ```
    fn set_popsize(mut self, popsize: i32) -> Result<Self> {
        self.popsize = popsize;
        self.update_dependent_params();
        Ok(self)
    }
    /// Sets origin of search
    ///
    /// Example
    ///
    /// ```rust
    /// use haru_cmaes::{CmaesParams, CmaesParamsValidator};
    ///
    /// let params = CmaesParams::new()
    ///     .and_then(|p| p.set_xstart(vec![0.0; 60]));
    ///
    /// assert!(params.is_ok());
    /// ```
    fn set_xstart(mut self, xstart: Vec<f32>) -> Result<Self::Output> {
        self.xstart = xstart;
        self.update_dependent_params();
        Ok(self)
    }

    /// Sets step size (sigma)
    ///
    /// Example
    ///
    /// ```rust
    /// use haru_cmaes::{CmaesParams, CmaesParamsValidator};
    ///
    /// let params = CmaesParams::new()
    ///     .and_then(|p| p.set_sigma(0.85));
    ///
    /// assert!(params.is_ok());
    /// ```
    fn set_sigma(mut self, sigma: f32) -> Result<Self::Output> {
        self.sigma = sigma;
        self.update_dependent_params();
        Ok(self)
    }
}
