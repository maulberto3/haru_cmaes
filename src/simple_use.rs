use crate::{
    fitness::{FitnessEvaluator, SquareAndSum},
    params::CmaesParams,
    state::{CmaesState, CmaesStateLogic},
    strategy::{Cmaes, CmaesOptimizer},
};
use anyhow::Result;

#[allow(unused_imports)]
use blas_src;

/// Example usage of the CMA-ES algorithm.
///
/// This function demonstrates a basic workflow of the CMA-ES
/// optimization algorithm using predefined parameters and fitness
/// function.
///
/// It initializes the CMA-ES algorithm, iterates
/// through a fixed number of generations, and prints the
/// average fitness of the best solutions.
///
/// Final solution is under state.best_y and state.best_y_fit
pub fn example() -> Result<()> {
    // Initialize CMA-ES parameters
    let params = CmaesParams {
        popsize: 50,
        xstart: vec![0.0; 50],
        sigma: 0.75,
    };

    // Create a new CMA-ES instance
    let cmaes = Cmaes::new(&params)?;

    // Initialize the CMA-ES state
    let mut state = CmaesState::init_state(&params)?;

    // Run the CMA-ES algorithm for 150 iterations
    for _i in 0..150 {
        // Generate a new population
        let mut pop = cmaes.ask(&mut state)?;

        // Evaluate the fitness of the population
        let mut fitness = SquareAndSum.evaluate(&pop)?;

        // Update the state with the new population and fitness values
        state = cmaes.tell(state, &mut pop, &mut fitness)?;
    }

    // Print the average fitness of the best solutions
    println!("Fitness (mean): {:+.4?}", &state.best_y_fit.mean());

    Ok(())
}
