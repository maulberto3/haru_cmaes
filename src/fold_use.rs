use crate::fitness::{FitnessEvaluator, MinOrMax};
use crate::objectives::SquareAndSum;
use crate::params::{CmaesParams, CmaesParamsValidator};
use crate::state::{CmaesState, CmaesStateLogic};
use crate::strategy::{CmaesAlgo, CmaesAlgoOptimizer};
use anyhow::Result;
// use nalgebra::DMatrix;
use std::env::var;
#[allow(unused_imports)]
use std::io::{self, Write};
use std::time::Instant;

pub fn fold_example() -> Result<CmaesState> {
    // Define verbose or not
    let verbose = var("VERBOSE").unwrap_or("No".to_string());

    // Take start time
    let start = Instant::now();

    // Create cost function, i.e. see fitness.rs for an example
    let obj_func = SquareAndSum {
        obj_dim: 50,
        dir: MinOrMax::Min,
        // target: 1.0,
        // output_dim: 2,
        // input_dim: 2,
        // data: DMatrix::from_row_slice(3, 4, &vec![
        //     1.2, 3.0, 2.0, 2.5,
        //     4.0, 5.5, 6.0, 2.5,
        //     6.0, 8.5, 7.0, 9.5,
        // ])
    };

    // Initialize CMA-ES parameters
    let params = CmaesParams::new()?
        .set_popsize(50)?
        .set_xstart(obj_func.evaluator_dim()?, 0.5)?
        .set_sigma(0.5)?
        .set_only_diag(true)?
        // If you set specific number of generations, you can scan through
        .set_num_gens(100)?;

    // Create a new CMA-ES instance
    let cmaes = CmaesAlgo::new(params)?;

    // Initialize the CMA-ES state
    let state = CmaesState::init_state(&cmaes.params)?;

    // SCAN the CMA-ES algorithm until close to objective value
    let state = cmaes.rollout_fold(state, obj_func)?;

    // Print the average fitness of the best solutions
    if verbose != "No" {
        println!();
        println!(
            "Fitness: {:+.5?} | Duration p/step: {:.5} secs",
            &state.best_y_fit.row(0)[0],
            (start.elapsed().as_micros() as f32) / 1000000.0 / (cmaes.params.num_gens as f32)
        );
        // dbg!(state);
        println!("{:+.5?}", &state.best_y);
    }

    Ok(state)
}
