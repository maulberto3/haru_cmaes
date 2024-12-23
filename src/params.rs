use anyhow::{anyhow, Result};
use ndarray::{s, Array1};

// TODO: try and use builder pattern

/// Parameters for CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
#[derive(Debug, Clone)]
pub struct CmaesParams {
    pub popsize: i32,           // Population size.
    pub xstart: Vec<f32>,       // Initial solution vector.
    pub sigma: f32,             // Step-size (standard deviation).
    pub tol: Option<f32>,       // Tolerance for convergence, optional
    pub obj_value: Option<f32>, // Known objective value, optional
    pub zs: Option<f32>,        // Enforce zero sparsity for quicker computational results, optional
}

/// Validated parameters for CMA-ES.
#[derive(Debug, Clone)]
pub struct CmaesParamsValid {
    pub popsize: i32,           // Population size
    pub xstart: Vec<f32>,       // Initial guess (mean vector)
    pub sigma: f32,             // Step-size (standard deviation)
    pub tol: Option<f32>,       // Tolerance for convergence, optional
    pub obj_value: Option<f32>, // Known objective value, optional
    pub zs: Option<f32>,        // Enforce zero sparsity for quicker computational results, optional
    pub n: f32,                 // Dimension of the problem space (xstart size)
    pub mu: i32,                // Number of parents (best individuals)
    pub weights: Array1<f32>,   // Weights for recombination
    pub mueff: f32,             // Effective number of parents
    pub cc: f32,                // Cumulation constant for the rank-one update
    pub cs: f32,                // Cumulation constant for the rank-mu update
    pub c1: f32,                // Learning rate for the rank-one update
    pub cmu: f32,               // Learning rate for the rank-mu update
    pub damps: f32,             // Damping for step-size adaptation
                                // pub lazy_gap_evals: f32, // Gap to postpone eigendecomposition
}

/// Trait defining validation methods for CMA-ES parameters.
pub trait CmaesParamsValidator {
    // Type for the validated parameters.
    type ValidatedParams;

    fn validate(params: CmaesParams) -> Result<Self::ValidatedParams>;
    fn validate_params(params: CmaesParams) -> Result<CmaesParams>;
    fn check_xstart(params: CmaesParams) -> Result<()>;
    fn check_popsize(params: CmaesParams) -> Result<()>;
    fn check_sigma(params: CmaesParams) -> Result<()>;

    fn add_default_params(params: CmaesParams) -> Result<Self::ValidatedParams>;
}

/// Trait for Validated Cmaes Params
impl CmaesParamsValidator for CmaesParamsValid {
    /// Type for the validated parameters.
    type ValidatedParams = CmaesParamsValid;

    /// Validates the provided parameters and returns a validated parameter set.
    fn validate(params: CmaesParams) -> Result<Self::ValidatedParams> {
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
        let params = CmaesParamsValid::add_default_params(params)?;
        Ok(params)
    }

    /// Validates the provided parameters to ensure they meet the constraints.
    fn validate_params(params: CmaesParams) -> Result<CmaesParams> {
        CmaesParamsValid::check_popsize(params.clone())?;
        CmaesParamsValid::check_xstart(params.clone())?;
        CmaesParamsValid::check_sigma(params.clone())?;
        Ok(params)
    }

    /// Checks if the `xstart` parameter meets its constraints.
    fn check_xstart(params: CmaesParams) -> Result<()> {
        if params.xstart.len() <= 1 {
            return Err(anyhow!("==> number of dimensions must be > 1."));
        }
        Ok(())
    }

    /// Checks if the `popsize` parameter meets its constraints.
    fn check_popsize(params: CmaesParams) -> Result<()> {
        if params.popsize <= 5 {
            return Err(anyhow!("==> popsize must be > 5."));
        }
        Ok(())
    }

    /// Checks if the `sigma` parameter meets its constraints.
    fn check_sigma(params: CmaesParams) -> Result<()> {
        if params.sigma <= 0.0 {
            return Err(anyhow!("==> sigma must be greater than 0.0."));
        }
        Ok(())
    }

    /// Creates default parameters for the CMA-ES algorithm based on the provided parameters.
    fn add_default_params(params: CmaesParams) -> Result<Self::ValidatedParams> {
        let popsize = params.popsize;
        let xstart = params.xstart;
        let sigma = params.sigma;

        let tol = params.tol;
        let obj_value = params.obj_value;

        let zs = params.zs;

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
            // Required
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
}
