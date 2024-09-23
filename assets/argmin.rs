use argmin::core::{CostFunction, Error, Executor};
use argmin::solver::particleswarm::ParticleSwarm;

// Define the Sphere function (sum of squares)
struct Sphere {}

impl CostFunction for Sphere {
    // Parameter type: vector of f64
    type Param = Vec<f64>;
    // Output type: fitness value (f64)
    type Output = f64;

    // Sum of squares (fitness function)
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        Ok(param.iter().map(|x| x.powi(2)).sum())
    }
}

fn run() -> Result<(), Error> {
    // Create the Sphere fitness function
    let cost_function = Sphere {};

    // Set up the bounds for the PSO algorithm
    let dim = 50;
    let lower_bound = vec![-1.0; dim];  // Lower bound for each dimension
    let upper_bound = vec![1.0; dim];   // Upper bound for each dimension

    // Initialize the PSO solver with population size of 50
    let solver = ParticleSwarm::new((lower_bound, upper_bound), 50);

    // Set up the executor with the cost function and solver
    let res = Executor::new(cost_function, solver)
        .configure(|state| state.max_iters(150))  // Run for 150 iterations
        .run()?;

    // Print the result (best solution and fitness value)
    println!("{res}");

    Ok(())
}

pub fn example() {
    // Run the PSO algorithm and handle errors
    if let Err(ref e) = run() {
        println!("{e}");
    }
}
