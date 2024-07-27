use cmaes::{CMAESOptions, DVector};
use anyhow::Result;

/// Example usage of the CMA-ES algorithm with the pengown crate.
///
/// This function demonstrates a basic workflow of the CMA-ES optimization algorithm using
/// predefined parameters and fitness function. It initializes the CMA-ES algorithm, iterates
/// through a fixed number of generations, and prints the average fitness of the best solutions.
///
/// # Returns
/// - `Result<()>`: Returns `Ok(())` if the function completes successfully, or an error if any
///   operation fails.
pub fn example() -> Result<()> {
    // Define the fitness function
    let square_and_sum = |x: &DVector<f64>| x.iter().map(|xi| xi.powi(2)).sum();

    // Set dimension and parameters
    let dim = 50;
    let popsize = 50;
    let sigma = 0.75;
    let mut cmaes_state = CMAESOptions::new(vec![1.0; dim], 1.0)
        .population_size(popsize)
        .initial_mean(vec![0.0; dim])
        .initial_step_size(sigma)
        .max_generations(150)
        .enable_printing(9999)
        .build(square_and_sum)
        .unwrap();
    
    // Run the CMA-ES algorithm for 150 iterations
    let _ = cmaes_state.run();
    
    // Here you would update your state and population if needed
    // For demonstration purposes, we're printing the best fitness value
    // println!("Best solution fitness: {:.4}", solution.0);

    Ok(())
}