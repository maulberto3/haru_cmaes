use crate::fitness::{allow_objective_func, FitnessEvaluator, Rastrigin};
use crate::params::{CmaesParams, CmaesParamsValidator};
use crate::state::{CmaesState, CmaesStateLogic};
use crate::strategy::{CmaesAlgo, CmaesAlgoOptimizer};
use anyhow::Result;
use nalgebra::DVector;
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
// Here, it is shown that if the mean of last 50 best
// optimization values is close to current best,
// the lops breaks
//
// Finally, the solution is under state.best_y and state.best_y_fit

pub fn example() -> Result<()> {
    // Define verbose or not
    let verbose = true;
    // let stdout = io::Stdout::

    // Take start time
    let start = Instant::now();

    // Create cost function, i.e. see fitness.rs for an example
    let objective_function = Rastrigin { obj_dim: 50 };
    // let objective_function = StdAndSum { obj_dim: 45 };

    // Check allowed objective function
    let (obj, obj_dim) = allow_objective_func(objective_function)?;

    // Initialize CMA-ES parameters
    let params = CmaesParams::new()?
        .set_popsize(50)?
        .set_xstart(vec![0.0; obj_dim])?
        // .set_xstart(Array1::random(obj_dim, Uniform::new(-0.5, 0.5)).to_vec())?
        .set_sigma(0.25)?;

    // Create a new CMA-ES instance
    let cmaes = CmaesAlgo::new(params)?;

    // Initialize the CMA-ES state
    let mut state = CmaesState::init_state(&cmaes.params)?;

    // Run the CMA-ES algorithm until close to objective value
    let (mut step, mut best_y) = (0, vec![99.]);
    loop {
        // Generate a new population
        let mut pop = cmaes.ask(&mut state)?;

        // Evaluate the fitness of the population
        let mut fitness = obj.evaluate(&pop)?;

        // Update the state with the new population and fitness values
        state = cmaes.tell(state, &mut pop, &mut fitness)?;

        // Save current best
        best_y.push(state.best_y_fit.row(0)[0]);

        // Stopping criteria 1: Check if we are there yet?
        let last_50_best_y = if best_y.len() > 25 {
            best_y[best_y.len() - 25..].to_vec()
        } else {
            best_y[..].to_vec()
        };
        let best_y_avg = DVector::from_vec(last_50_best_y).mean();
        if verbose {
            print!("{:+.4?}  ", &best_y_avg);
            io::stdout().flush().unwrap()
        }
        ////////////////
        if (state.best_y_fit.row(0)[0] - best_y_avg).abs() < cmaes.params.tol {
            if verbose {
                println!("  ===> Search stopped due to tolerance change met")
            }
            break;
        }

        step += 1;
    }
    // Print the average fitness of the best solutions
    if verbose {
        println!(
            "Step {} | Fitness: {:+.4?} | Duration p/step: {:.4} secs",
            step,
            &state.best_y_fit.row(0)[0],
            (start.elapsed().as_micros() as f32) / 1000000.0 / (step as f32)
        )
    }

    Ok(())
}
