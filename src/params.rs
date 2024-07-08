use anyhow::{anyhow, Result};
use ndarray::{s, Array1};
use ndarray_linalg::Scalar;
// use ndarray_stats::QuantileExt;

#[derive(Debug, Clone)]
pub struct CmaesParams {
    // Required
    pub popsize: i32,
    pub xstart: Vec<f32>,
    pub sigma: f32,
}

#[derive(Debug, Clone)]
pub struct CmaesParamsValid {
    pub popsize: i32,
    pub xstart: Vec<f32>,
    pub sigma: f32,
    pub n: f32,
    pub chin: f32,
    pub mu: f32,
    pub weights: Array1<f32>,
    pub mueff: f32,
    pub cc: f32,
    pub cs: f32,
    pub c1: f32,
    pub cmu: f32,
    pub damps: f32,
    pub lazy_gap_evals: f32,
}

impl CmaesParamsValid {
    pub fn validate(params: &CmaesParams) -> Result<CmaesParamsValid> {
        print!("Validating initial parameters... ");
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
        println!("Done.");

        print!("Computing default parameters... ");
        let params = CmaesParamsValid::create_default_params(params)?;
        println!("Done.");
        Ok(params)
    }

    fn create_default_params(params: CmaesParams) -> Result<CmaesParamsValid> {
        let popsize = params.popsize;
        let xstart = params.xstart;
        let sigma = params.sigma;

        let n = xstart.len() as f32;
        let chin = n.sqrt() * (1. - 1. / (4. * n) + 1. / (21. * n.square()));
        let mu = (popsize / 2) as f32;

        let _iterable: Vec<f32> = (0..popsize as i32)
            .enumerate()
            .map(|(_x, i)| {
                if i < (mu as i32) {
                    ((popsize / 2) as f32 + 0.5).ln() - ((i + 1) as f32).ln()
                } else {
                    0.0
                }
            })
            .collect();
        let _weights: Array1<f32> = Array1::from_iter(_iterable);
        let _w_sum = _weights.slice(s![..(mu as i32)]).sum();
        let weights: Array1<f32> = _weights.mapv(|x| x / _w_sum);
        let mueff: f32 = _weights.slice(s![..(mu as i32)]).sum().square()
            / _weights.slice(s![..(mu as i32)]).mapv(|x| x.square()).sum();

        // time constant for cumulation for C
        let cc = (4. + mueff / n) / (n + 4. + 2. * mueff / n);
        // time constant for cumulation for sigma control
        let cs = (mueff + 2.) / (n + mueff + 5.);
        // learning rate for rank-one update of C
        let c1 = 2. / (n + 1.3).square() + mueff;
        // and for rank-mu update
        let cmu = (1. - c1).min(2. * (mueff - 2. + 1. / mueff) / (n + 2.).square() + mueff);
        // damping for sigma, usually close to 1
        let damps = 2. * mueff / popsize as f32 + 0.3 + cs;

        // gap to postpone eigendecomposition to achieve O(N**2) per eval
        // 0.5 is chosen such that eig takes 2 times the time of tell in >=20-D
        let lazy_gap_evals = 0.5 * n * (popsize as f32) * (c1 + cmu).powi(-1) / n.square();

        let valid_params = CmaesParamsValid {
            popsize,
            xstart,
            sigma,
            n,
            chin,
            mu,
            weights,
            mueff,
            cc,
            cs,
            c1,
            cmu,
            damps,
            lazy_gap_evals,
        };
        Ok(valid_params)
    }

    fn validate_params(params: &CmaesParams) -> Result<CmaesParams> {
        CmaesParamsValid::check_popsize(params)?;
        CmaesParamsValid::check_xstart(params)?;
        CmaesParamsValid::check_sigma(params)?;
        Ok(params.clone())
    }

    fn check_xstart(params: &CmaesParams) -> Result<()> {
        if params.xstart.len() <= 1 {
            return Err(anyhow!("==> number of dimensions must be > 1."));
        }
        Ok(())
    }

    fn check_popsize(params: &CmaesParams) -> Result<()> {
        if params.popsize <= 5 {
            return Err(anyhow!("==> popsize must be > 5."));
        }
        Ok(())
    }

    fn check_sigma(params: &CmaesParams) -> Result<()> {
        if params.sigma <= 0.0 {
            return Err(anyhow!("==> sigma must be greater than 0.0."));
        }
        Ok(())
    }

    // fn calculate_popsize(num_dims: &f32) -> Result<i32> {
    //     Ok(4 + (3.0 * (*num_dims).ln()).floor() as i32)
    // }
}
