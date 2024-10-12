use crate::{
    CmaesParams, 
    state::{CmaesState, CmaesStateLogic}, 
    strategy::{CmaesAlgo, CmaesOptimizer}, 
    fitness::{FitnessEvaluator, SquareAndSum}, 
};
use anyhow::Result;

// #[allow(unused_imports)]
// use blas_src;

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
/// Notice you have to create your own Objective function
/// here it is SquareAndSum that implements FitnessEvaluator trait
///
/// Final solution is under state.best_y and state.best_y_fit
pub fn example() -> Result<()> {
    // Initialize CMA-ES parameters
    let params = CmaesParams {
        popsize: 50,
        xstart: vec![0.0; 50],
        sigma: 0.75,
        tol: Some(0.0001),
        obj_value: Some(0.0),
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
        let mut fitness = SquareAndSum.evaluate(&pop)?;

        // Update the state with the new population and fitness values
        state = cmaes.tell(state, &mut pop, &mut fitness)?;
        
        // Manual check
        if let Some(obj_value) = cmaes.validated_params.obj_value.as_ref() {
            if let Some(tol) = cmaes.validated_params.tol.as_ref() {
                let curr = state.best_y.first().unwrap();
                if (curr - obj_value).abs() < *tol {
                    break
                }
            }
        }
        step += 1
    }
    // Print the average fitness of the best solutions
    println!("Step {} | Fitness: {:+.4?}", step, &state.best_y.first().unwrap());

    Ok(())
}
