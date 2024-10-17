// use anyhow::Result;
// use nlopt::{Algorithm, Nlopt, Target};

// /// Example usage of the Cobyla algorithm with the nlopt crate.
// ///
// /// This function demonstrates a basic workflow of the Nelder-Mead optimization algorithm using
// /// predefined parameters and fitness function. It initializes the optimizer, sets up the problem,
// /// and prints the final solution.
// ///
// /// # Returns
// /// - `Result<()>`: Returns `Ok(())` if the function completes successfully, or an error if any
// ///   operation fails.
// pub fn example() -> Result<()> {
//     // Define the fitness function
//     fn square_and_sum(x: &[f64], _grad: Option<&mut [f64]>, _param: &mut ()) -> f64 {
//         x.iter().map(|xi| xi.powi(2)).sum()
//     }
    
//     // Set dimension and parameters
//     let dim = 50;
//     let max_iterations = 150; // Maximum number of iterations

//     // Create an optimizer for the Nelder-Mead algorithm
//     let mut optimizer = Nlopt::new(
//         Algorithm::Cobyla, // Optimization algorithm
//         dim,                   // Dimension of the problem
//         square_and_sum,        // Objective function (fitness)
//         Target::Minimize,      // Target to minimize the function
//         (),                    // Optional user data
//     );
    

//     // Set initial guess for the optimization
//     let mut x = vec![1.0; dim];

//     // Set relative tolerance on optimization
//     let _ = optimizer.set_xtol_rel(1e-4);

//     // Set the maximum number of iterations
//     let _ = optimizer.set_maxeval(max_iterations);

//     // Run the optimization
//     let result = optimizer.optimize(&mut x);

//     // Print the best solution and its fitness value
//     println!("Best solution fitness: {:?}", result);
//     println!("Best solution: {:?}", x);

//     Ok(())
// }
