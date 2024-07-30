use anyhow::{anyhow, Result};
use ndarray::{s, Array1};

/// Parameters for CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
#[derive(Debug, Clone)]
pub struct CmaesParams {
    pub popsize: i32,       // Population size.
    pub xstart: Vec<f32>,   // Initial solution vector.
    pub sigma: f32,         // Step-size (standard deviation).
}

/// Validated parameters for CMA-ES.
#[derive(Debug, Clone)]
pub struct CmaesParamsValid {
    pub popsize: i32,
    pub xstart: Vec<f32>,
    pub sigma: f32,
    pub n: f32,
    // pub chin: f32,
    pub mu: i32,
    pub weights: Array1<f32>,
    pub mueff: f32,
    pub cc: f32,
    pub cs: f32,
    pub c1: f32,
    pub cmu: f32,
    pub damps: f32,
    // pub lazy_gap_evals: f32,
}

/// Trait defining validation methods for CMA-ES parameters.
pub trait CmaesParamsValidator {
    /// Type for the validated parameters.
    type ValidatedParams;

    fn validate(params: &CmaesParams) -> Result<Self::ValidatedParams>;
    fn validate_params(params: &CmaesParams) -> Result<CmaesParams>;
    fn check_xstart(params: &CmaesParams) -> Result<()>;
    fn check_popsize(params: &CmaesParams) -> Result<()>;
    fn check_sigma(params: &CmaesParams) -> Result<()>;
    fn create_default_params(params: CmaesParams) -> Result<Self::ValidatedParams>;
}

impl CmaesParamsValidator for CmaesParamsValid {
    /// Type for the validated parameters.
    type ValidatedParams = CmaesParamsValid;

    /// Validates the provided parameters and returns a validated parameter set.
    ///
    /// # Errors
    /// Returns an error if any of the initial parameters do not meet their constraints.
    fn validate(params: &CmaesParams) -> Result<Self::ValidatedParams> {
        // print!("Validating initial parameters... ");
        let params = match CmaesParamsValid::validate_params(params) {
            Ok(params) => params,
            Err(e) => {
                eprintln!(
                    "An initial Cmaes parameter is not following its constraint: {}",
                    e
                );
                return Err(e);
            }
        };

        // print!("Computing default parameters... ");
        let params = CmaesParamsValid::create_default_params(params)?;
        Ok(params)
    }

    /// Creates default parameters for the CMA-ES algorithm based on the provided parameters.
    fn create_default_params(params: CmaesParams) -> Result<Self::ValidatedParams> {
        let popsize = params.popsize;
        let xstart = params.xstart;
        let sigma = params.sigma;

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

        let valid_params = CmaesParamsValid {
            popsize,
            xstart,
            sigma,
            n,
            // chin,
            mu,
            weights,
            mueff,
            cc,
            cs,
            c1,
            cmu,
            damps,
            // lazy_gap_evals,
        };
        Ok(valid_params)
    }

    // /// Validates the provided parameters to ensure they meet the constraints.
    // ///
    // /// # Errors
    // /// Returns an error if any parameter does not meet its constraint.
    fn validate_params(params: &CmaesParams) -> Result<CmaesParams> {
        CmaesParamsValid::check_popsize(params)?;
        CmaesParamsValid::check_xstart(params)?;
        CmaesParamsValid::check_sigma(params)?;
        Ok(params.clone())
    }

    /// Checks if the `xstart` parameter meets its constraints.
    ///
    /// # Errors
    /// Returns an error if the number of dimensions is not greater than 1.
    fn check_xstart(params: &CmaesParams) -> Result<()> {
        if params.xstart.len() <= 1 {
            return Err(anyhow!("==> number of dimensions must be > 1."));
        }
        Ok(())
    }

    /// Checks if the `popsize` parameter meets its constraints.
    ///
    /// # Errors
    /// Returns an error if the population size is not greater than 5.
    fn check_popsize(params: &CmaesParams) -> Result<()> {
        if params.popsize <= 5 {
            return Err(anyhow!("==> popsize must be > 5."));
        }
        Ok(())
    }

    /// Checks if the `sigma` parameter meets its constraints.
    ///
    /// # Errors
    /// Returns an error if the step-size is not greater than 0.
    fn check_sigma(params: &CmaesParams) -> Result<()> {
        if params.sigma <= 0.0 {
            return Err(anyhow!("==> sigma must be greater than 0.0."));
        }
        Ok(())
    }
}
