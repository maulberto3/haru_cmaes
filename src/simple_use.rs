use std::time::Instant;

use crate::{
    fitness::{allow_objective_func, FitnessEvaluator, SquareAndSum},
    state::{CmaesState, CmaesStateLogic},
    strategy::{CmaesAlgo, CmaesOptimizer},
    CmaesParams,
};
use anyhow::Result;

// #[allow(unused_imports)]
// use blas_src;

// Example usage of the CMA-ES algorithm.
//
// First, it checks that the objective function is allowed
//
// This function demonstrates a basic workflow of the CMA-ES
// optimization algorithm using predefined parameters and fitness
// function.
//
// It initializes the CMA-ES algorithm, iterates
// through a fixed number of generations, and prints the
// average fitness of the best solutions.
//
// Notice you have to create your own Objective function
// here it is SquareAndSum that implements FitnessEvaluator trait
//
// Final solution is under state.best_y and state.best_y_fit

pub fn example() -> Result<()> {
    // Take start time
    let start = Instant::now();

    // Check allowed objective function
    let obj = allow_objective_func(SquareAndSum)?;

    // Initialize CMA-ES parameters
    let params = CmaesParams {
        // Required
        popsize: 50,
        xstart: vec![0.0; 50],
        sigma: 0.75,
        // Optional
        tol: Some(0.0001),
        obj_value: Some(0.0), // This has to make sense for your objective function
    };

    // Create a new CMA-ES instance
    let cmaes = CmaesAlgo::new(params)?;

    // Initialize the CMA-ES state
    let mut state = CmaesState::init_state(&cmaes.validated_params)?;

    // Run the CMA-ES algorithm until close to objective value
    let mut step = 0;
    loop {
        // Generate a new population
        let mut pop = cmaes.ask(&mut state)?;

        // Evaluate the fitness of the population
        let mut fitness = obj.evaluate(&pop)?;

        // Update the state with the new population and fitness values
        state = cmaes.tell(state, &mut pop, &mut fitness)?;

        // Are we there yet?
        let obj_value = cmaes.validated_params.obj_value.as_ref();
        let tol = cmaes.validated_params.tol.as_ref();
        match (obj_value, tol) {
            (Some(obj_value), Some(tol)) => {
                let curr = state.best_y.first().unwrap();
                if (curr - obj_value).abs() < *tol {
                    // If we are close to obj_value less than tol, we are there (break)
                    break;
                }
            }
            _ => (),
        }
        step += 1;

    }
    // Print the average fitness of the best solutions
    println!(
        "Step {} | Fitness: {:+.4?} | Duration p/step: {:.4} secs",
        step,
        &state.best_y.first().unwrap(),
        (start.elapsed().as_micros() as f32) / 1000000.0 / (step as f32)
    );

    Ok(())
}
