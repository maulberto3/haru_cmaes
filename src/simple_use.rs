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

// #[allow(unused_imports)]
// use blas_src;

// Example usage of the CMA-ES algorithm.
//
// First, it defines and checks that your objective
// function is allowed
//
// Then, create default Cmaes parameters, after which
// can be adjusted to specific values
//
// Then, we pass those params to Cmaes algorithm and start
// the ask-tell loop
//
// Here, it is shown that if the mean of last 25 best
// optimization values is close to current best,
// the lops breaks
//
// Finally, the solution is under state.best_y and state.best_y_fit

pub fn example() -> Result<()> {
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
        .set_xstart(vec![0.5; obj_func.evaluator_dim()?])?
        .set_sigma(0.5)?;

    // Create a new CMA-ES instance
    let cmaes = CmaesAlgo::new(params)?;

    // Initialize the CMA-ES state
    let mut state = CmaesState::init_state(&cmaes.params)?;

    // Run the CMA-ES algorithm until close to objective value
    let mut step = 0;
    loop {
        // Generate a new population
        let mut pop = cmaes.ask(&mut state)?;

        // Evaluate the fitness of the population
        let mut fitness = obj_func.evaluate(&pop)?;

        // Update the state with the new population and fitness values
        state = cmaes.tell(state, &mut pop, &mut fitness)?;

        // Continue or done?
        if let Ok(true) = cmaes.is_done(&state, step) {
            break;
        }

        // Log some info
        if verbose != "No" {
            print!("{:+.5?} ", &state.best_y_fit.row(0)[0]);
            // print!("best y {:+.5?} ", &state.best_y.row(0)[0]);
            io::stdout().flush().unwrap()
        }

        step += 1;
    }
    // Print the average fitness of the best solutions
    if verbose != "No" {
        println!(
            "Step {} | Fitness: {:+.5?} | Duration p/step: {:.5} secs",
            step,
            &state.best_y_fit.row(0)[0],
            (start.elapsed().as_micros() as f32) / 1000000.0 / (step as f32)
        );
        // dbg!(state);
        println!("{:+.5?}", &state.best_y);
    }

    Ok(())
}
