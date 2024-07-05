use anyhow::{anyhow, Result};
use ndarray::{s, Array1};
use ndarray_stats::QuantileExt;

#[derive(Debug, Clone)]
pub struct CmaesParams {
    // Required
    pub mean: Vec<f32>,
    pub sigma: f32,
    pub popsize: i32,
    // Optional
    // pub bounds
    // pub n_max_resampling: Option<i32>,
    // pub seed: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct CmaesParamsValid {
    pub mean: Array1<f32>,
    pub sigma: f32,
    pub num_dims: f32,
    pub popsize: i32,
    pub mu: usize,
    pub weights_prime: Array1<f32>,
    pub mu_eff: f32,
    pub mu_eff_rest: f32,
    pub cm: f32,
    pub c_sigma: f32,
    pub d_sigma: f32,
    pub chi_n: f32,
    pub cc: f32,
    pub weights: Array1<f32>,
    pub c1: f32,
    pub cmu: f32,
}

impl CmaesParamsValid {
    pub fn validate(params: &CmaesParams) -> Result<CmaesParamsValid> {
        print!("Computing default parameters... ");
        let params_ = CmaesParamsValid::create_default_params(params)?;
        println!("Done.");

        print!("Validating initial parameters... ");
        match CmaesParamsValid::validate_params(&params_) {
            Ok(_) => {
                println!(" Done.");
                Ok(params_)
            }
            Err(e) => {
                eprint!("An initial Cmaes parameter is not following its constraint: ");
                eprintln!("{} \n", e);
                panic!();
            }
        }
    }

    fn create_default_params(params: &CmaesParams) -> Result<CmaesParamsValid> {
        let mean: Array1<f32> = Array1::from_vec(params.mean.clone());
        let sigma: f32 = params.sigma;
        let num_dims: f32 = params.mean.len() as f32;
        let popsize: i32 = if params.popsize <= 5 {
            println!("Parameter popsize smaller than 5, recalculating default value.");
            CmaesParamsValid::calculate_popsize(&num_dims)?
        } else {
            params.popsize
        };
        let mu = (params.popsize / 2) as usize;
        let weights_prime: Array1<f32> = Array1::from_vec(
            (0..popsize)
                .map(|i| (((popsize as f32 + 1.0) / 2.0).ln() - (i as f32 + 1.0).ln()))
                .collect(),
        );
        let sum_weights_prime: f32 = weights_prime.slice(s![..mu]).sum();
        let sum_weights_prime_squared: f32 = weights_prime.slice(s![..mu]).mapv(|w| w * w).sum();
        let mu_eff = (sum_weights_prime * sum_weights_prime) / sum_weights_prime_squared;

        let sum_weights_prime_minus: f32 = weights_prime.slice(s![mu..]).sum();
        let sum_weights_prime_minus_squared: f32 =
            weights_prime.slice(s![mu..]).mapv(|w| w * w).sum();
        let mu_eff_rest =
            (sum_weights_prime_minus * sum_weights_prime_minus) / sum_weights_prime_minus_squared;

        let cm = 1.0;

        let c_sigma = (mu_eff + 2.0) / (num_dims + mu_eff + 5.0);
        let d_sigma = 1.0
            + 2.0
                * CmaesParamsValid::max_f32(0.0, ((mu_eff - 1.0).sqrt() / (num_dims + 1.0)) - 1.0)
            + c_sigma;

        let chi_n = (num_dims).sqrt()
            * (1.0 - (1.0 / (4.0 * num_dims)) + 1.0 / (21.0 * (num_dims * num_dims)));

        let cc = (4.0 + mu_eff / num_dims) / (num_dims + 4.0 + 2.0 * mu_eff / num_dims);

        let positive_sum: f32 =
            weights_prime.fold(0.0, |acc, &x| if x > 0.0 { acc + x } else { acc });
        let negative_sum: f32 =
            weights_prime.fold(0.0, |acc, &x| if x < 0.0 { acc + x.abs() } else { acc });
        let alpha_cov = 2.0;
        let c1: f32 = alpha_cov / ((num_dims + 1.3).powi(2) + mu_eff);
        let cmu: Vec<f32> = vec![
            1.0 - c1 - f32::EPSILON,
            alpha_cov * (mu_eff - 2.0 + 1.0 / mu_eff)
                / ((num_dims + 2.0).powi(2) + alpha_cov * mu_eff / 2.0),
        ];
        let cmu: f32 = Array1::from_vec(cmu).min().unwrap().to_owned();
        let min_alpha: Vec<f32> = vec![
            1.0 + c1 / cmu,                             // eq.50
            1.0 + (2.0 * mu_eff_rest) / (mu_eff + 2.0), // eq.51
            (1.0 - c1 - cmu) / (num_dims * cmu),        // eq.52
        ];
        let min_alpha: f32 = Array1::from_vec(min_alpha).min().unwrap().to_owned();
        let weights: Array1<f32> = weights_prime.mapv(|x| {
            if x >= 0.0 {
                x / positive_sum
            } else {
                min_alpha * x / negative_sum
            }
        });

        // np.where(
        //     weights_prime >= 0,
        //     1 / positive_sum * weights_prime,
        //     min_alpha / negative_sum * weights_prime,
        // )

        let params_ = CmaesParamsValid {
            mean,
            sigma,
            num_dims,
            popsize,
            mu,
            weights_prime,
            mu_eff,
            mu_eff_rest,
            cm,
            c_sigma,
            d_sigma,
            chi_n,
            cc,
            weights,
            c1,
            cmu,
        };
        Ok(params_)
    }

    fn validate_params(params_: &CmaesParamsValid) -> Result<()> {
        CmaesParamsValid::check_mean_length(params_)?;
        CmaesParamsValid::check_sigma(params_)?;
        CmaesParamsValid::check_popsize(params_)?;
        Ok(())
    }

    fn check_mean_length(params_: &CmaesParamsValid) -> Result<()> {
        if params_.mean.len() <= 1 {
            return Err(anyhow!("==> number of dimensions must be > 1."));
        }
        Ok(())
    }

    fn check_sigma(params_: &CmaesParamsValid) -> Result<()> {
        if params_.sigma <= 0.0 {
            return Err(anyhow!("==> sigma must be > 0.0."));
        }
        Ok(())
    }

    fn check_popsize(params_: &CmaesParamsValid) -> Result<()> {
        if params_.popsize <= 5 {
            return Err(anyhow!("==> popsize must be > 5."));
        }
        Ok(())
    }

    fn calculate_popsize(num_dims: &f32) -> Result<i32> {
        Ok(4 + (3.0 * (*num_dims).ln()).floor() as i32)
    }

    fn max_f32(a: f32, b: f32) -> f32 {
        if a > b {
            a
        } else {
            b
        }
    }
}
