use anyhow::Result;
use nalgebra::DVector;

/// Parameters for CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
#[derive(Debug, Clone)]
pub struct CmaesParams {
    pub popsize: i32,          // Population size
    pub xstart: Vec<f32>,      // Initial guess (mean vector)
    pub num_gens: i32,         // Initially run for exact amount of generations
    pub sigma: f32,            // Step-size (standard deviation)
    pub tol: f32,              // Tolerance for convergence, optional
    pub only_diag: bool,       // Whether to use only diag and no covariances or not
    pub n: f32,                // Dimension of the problem space (xstart size)
    pub mu: i32,               // Number of parents (best individuals)
    pub weights: DVector<f32>, // Weights for recombination
    pub mueff: f32,            // Effective number of parents
    pub cc: f32,               // Cumulation constant for the rank-one update
    pub cs: f32,               // Cumulation constant for the rank-mu update
    pub c1: f32,               // Learning rate for the rank-one update
    pub cmu: f32,              // Learning rate for the rank-mu update
    pub damps: f32,            // Damping for step-size adaptation
                               // pub lazy_gap_evals: f32, // Gap to postpone eigendecomposition
}

/// Trait for CMA-ES parameters.
pub trait CmaesParamsValidator {
    type Validated;
    fn new() -> Result<Self::Validated>;
    // Fundamental
    fn set_popsize(self, popsize: i32) -> Result<Self::Validated>;
    fn set_xstart(self, capacity: usize, origin: f32) -> Result<Self::Validated>;
    fn set_sigma(self, sigma: f32) -> Result<Self::Validated>;
    // Helper
    fn update_dependent_params(&mut self);
    // Other worth specifying
    fn set_tol(self, tol: f32) -> Result<Self::Validated>;
    fn set_only_diag(self, only_diag: bool) -> Result<Self::Validated>;
    fn set_num_gens(self, num_gens: i32) -> Result<Self::Validated>;
}

/// Implmenting Trait for CMA-ES parameters.
impl CmaesParamsValidator for CmaesParams {
    type Validated = CmaesParams;

    /// Creates default parameters for the CMA-ES algorithm based on the provided parameters.
    ///
    /// ```rust
    /// use haru_cmaes::params::{CmaesParams, CmaesParamsValidator};
    ///
    /// let params = CmaesParams::new();
    ///
    /// assert!(params.is_ok());
    /// ```
    fn new() -> Result<Self::Validated> {
        // Must update all parameters if this one's setter is used
        let popsize: i32 = 10;
        // Must update all parameters if this one's setter is used
        let xstart = vec![0.0; 6];

        let sigma = 0.75;
        let tol = 0.001;
        let only_diag = false;
        let num_gens = 100;

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
        let weights: DVector<f32> = DVector::from_vec(iterable);
        let w_sum: f32 = weights.rows(0, mu as usize).iter().sum();
        let weights: DVector<f32> = weights.map(|x| x / w_sum);
        let weights_mu = weights.rows(0, mu as usize).into_owned();
        let mueff: f32 = (weights_mu.iter().sum::<f32>().powi(2)) / weights_mu.map(|x| x * x).sum();
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
            // Others
            tol,
            only_diag,
            num_gens,
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
        let weights: DVector<f32> = DVector::from_vec(iterable);
        let w_sum: f32 = weights.rows(0, self.mu as usize).iter().sum();
        self.weights = weights.map(|x| x / w_sum);
        let weights_mu = &self.weights.rows(0, self.mu as usize).into_owned();
        self.mueff = (weights_mu.iter().sum::<f32>().powi(2)) / weights_mu.map(|x| x * x).sum();
        self.cc = (4. + self.mueff / self.n) / (self.n + 4. + 2. * self.mueff / self.n);
        self.cs = (self.mueff + 2.) / (self.n + self.mueff + 5.);
        self.c1 = 2. / ((self.n + 1.3).powi(2) + self.mueff);
        self.cmu = (1. - self.c1)
            .min(2. * (self.mueff - 2. + 1. / self.mueff) / ((self.n + 2.).powi(2) + self.mueff));
        self.damps = 2. * self.mueff / k + 0.3 + self.cs;
    }

    /// Sets population size.
    ///
    /// ```rust
    /// use haru_cmaes::params::{CmaesParams, CmaesParamsValidator};
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
    /// Sets origin of search.
    ///
    /// ```rust
    /// use haru_cmaes::params::{CmaesParams, CmaesParamsValidator};
    ///
    /// let params = CmaesParams::new()
    ///     .and_then(|p| p.set_xstart(15, 0.75) );
    ///
    /// assert!(params.is_ok());
    /// ```
    fn set_xstart(mut self, capacity: usize, origin: f32) -> Result<Self::Validated> {
        self.xstart = vec![origin; capacity];
        self.update_dependent_params();
        Ok(self)
    }

    /// Sets step size (sigma).
    ///
    /// ```rust
    /// use haru_cmaes::params::{CmaesParams, CmaesParamsValidator};
    ///
    /// let params = CmaesParams::new()
    ///     .and_then(|p| p.set_sigma(0.85));
    ///
    /// assert!(params.is_ok());
    /// ```
    fn set_sigma(mut self, sigma: f32) -> Result<Self::Validated> {
        self.sigma = sigma;
        Ok(self)
    }

    /// Sets tolerance.
    ///
    /// ```rust
    /// use haru_cmaes::params::{CmaesParams, CmaesParamsValidator};
    ///
    /// let params = CmaesParams::new()
    ///     .and_then(|p| p.set_tol(0.1));
    ///
    /// assert!(params.is_ok());
    /// ```
    fn set_tol(mut self, tol: f32) -> Result<Self::Validated> {
        self.tol = tol;
        Ok(self)
    }

    /// Sets enforce covariance sparsity.
    ///
    /// ```rust
    /// use haru_cmaes::params::{CmaesParams, CmaesParamsValidator};
    ///
    /// let params = CmaesParams::new()
    ///     .and_then(|p| p.set_only_diag(true));
    ///
    /// assert!(params.is_ok());
    /// ```
    fn set_only_diag(mut self, only_diag: bool) -> Result<Self::Validated> {
        self.only_diag = only_diag;
        Ok(self)
    }

    /// Sets enforce covariance sparsity.
    ///
    /// ```rust
    /// use haru_cmaes::params::{CmaesParams, CmaesParamsValidator};
    ///
    /// let params = CmaesParams::new()
    ///     .and_then(|p| p.set_num_gens(150));
    ///
    /// assert!(params.is_ok());
    /// ```
    fn set_num_gens(mut self, num_gens: i32) -> Result<Self::Validated> {
        self.num_gens = num_gens;
        Ok(self)
    }
}
