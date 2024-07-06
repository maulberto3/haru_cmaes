use anyhow::{anyhow, Result};
use ndarray::{s, Array1};
use ndarray_linalg::Scalar;
use ndarray_stats::QuantileExt;

#[derive(Debug, Clone)]
pub struct CmaesParams {
    // Required
    pub num_dims: i32,
    pub popsize: i32,
}

#[derive(Debug, Clone)]
pub struct CmaesParamsValid {
    pub n: f32,
    pub popsize: i32,
    pub chin: f32,
    pub mu: f32,
    pub weights: Array1<f32>,
    pub mueff: f32,
    pub cc: f32,
    pub cs: f32,
    pub c1: f32,
    pub cmu: f32,
    pub damps: f32,
}

impl CmaesParamsValid {
    pub fn validate(params: &CmaesParams) -> Result<CmaesParamsValid> {
        print!("Validating initial parameters... ");
        let params = match CmaesParamsValid::validate_params(params) {
            Ok(params) => params,
            Err(e) => {
                eprintln!("An initial Cmaes parameter is not following its constraint: {} \n", e);
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
        let n = params.num_dims as f32;
        let popsize = params.popsize;
        let chin = n.sqrt() * (1. - 1. / (4. * n) + 1. / (21. * n.square()));
        let mu = (popsize  / 2) as f32;

        let _iterable: Vec<f32> = (0..popsize as i32).enumerate().map(|(_x, i)| {
            if i < (mu as i32) {
                ((popsize / 2) as f32 + 0.5).ln() - ((i + 1) as f32).ln()
            } else {
                0.0
            }
        }).collect();
        let _weights: Array1<f32> = Array1::from_iter(_iterable);
        let _w_sum = _weights.slice(s![..(mu as i32)]).sum();
        let weights: Array1<f32> = _weights.mapv(|x| x / _w_sum);
        let mueff: f32 = _weights.slice(s![..(mu as i32)]).sum().square() / _weights.slice(s![..(mu as i32)]).mapv(|x| x.square()).sum();

        // time constant for cumulation for C
        let cc = (4. + mueff / n) / ( n + 4. + 2. * mueff / n);
        // // time constant for cumulation for sigma control
        let cs = ( mueff + 2. ) / ( n + mueff + 5.);
        // // learning rate for rank-one update of C
        let c1 = 2. / ( n + 1.3 ).square() + mueff;
        // // and for rank-mu update
        let cmu = ( 1. - c1 ).min( 2. * ( mueff - 2. + 1. / mueff ) / ( n + 2.).square() + mueff );
        // // damping for sigma, usually close to 1
        let damps = 2. * mueff / popsize as f32 + 0.3 + cs;

        let valid_params = CmaesParamsValid { 
            n, 
            popsize,
            chin,
            mu,
            weights,
            mueff,
            cc,
            cs,
            c1,
            cmu,
            damps,
        };
        Ok(valid_params)
    }

    fn validate_params(params: &CmaesParams) -> Result<CmaesParams> {
        CmaesParamsValid::check_num_dims(params)?;
        CmaesParamsValid::check_popsize(params)?;
        Ok(params.clone())
    }

    fn check_num_dims(params: &CmaesParams) -> Result<()> {
        if params.num_dims <= 1 {
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

    // fn calculate_popsize(num_dims: &f32) -> Result<i32> {
    //     Ok(4 + (3.0 * (*num_dims).ln()).floor() as i32)
    // }
}
