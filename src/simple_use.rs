use std::time::Instant;

use crate::{
    fitness::{allow_objective_func, FitnessEvaluator, SquareAndSum, StdAndSum},
    CmaesAlgo, CmaesAlgoOptimizer, CmaesParams, CmaesParamsValidator, CmaesState, CmaesStateLogic,
};
use anyhow::Result;
use ndarray::Array1;

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

    // Create cost function, i.e. see fitness.rs for an example
    let objective_function = SquareAndSum { obj_dim: 40 };
    // let objective_function = StdAndSum { obj_dim: 45 };

    // Check allowed objective function
    let (obj, obj_dim) = allow_objective_func(objective_function)?;

    // Initialize CMA-ES parameters
    let params = CmaesParams::new()?
        .set_popsize(20)?
        .set_xstart(vec![0.0; obj_dim])?
        .set_sigma(0.75)?;

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
        best_y.push(*state.best_y_fit.first().unwrap());

        // Stopping criteria 1: Check if we are there yet?
        let last_50_best_y = if best_y.len() > 50 {
            best_y[best_y.len() - 50..].to_vec()
        } else {
            best_y[..].to_vec()
        };
        let best_y_avg = Array1::from_vec(last_50_best_y).mean().unwrap();
        ////////////////
        // TODO
        // Allow flag for verbose state
        print!("{:+.4?}  ", &best_y_avg);
        ////////////////
        if (state.best_y_fit.first().unwrap() - best_y_avg).abs() < cmaes.params.tol {
            println!("Search stopped due to closeness to target");
            break;
        }    

        step += 1;
    }
    // Print the average fitness of the best solutions
    println!(
        "Step {} | Fitness: {:+.4?} | Duration p/step: {:.4} secs",
        step,
        &state.best_y_fit.first().unwrap(),
        (start.elapsed().as_micros() as f32) / 1000000.0 / (step as f32)
    );

    Ok(())
}
