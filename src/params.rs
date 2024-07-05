use anyhow::{anyhow, Result};
use ndarray::{s, Array1};
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

        let valid_params = CmaesParamsValid { n, popsize };
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
