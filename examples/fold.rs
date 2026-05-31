use haru_cmaes::fitness::{MinOrMax, UserFitness};
use haru_cmaes::params::{CmaesParams, CmaesParamsValidator};
use haru_cmaes::state::{CmaesState, CmaesStateLogic};
use haru_cmaes::strategy::{CmaesAlgo, CmaesAlgoOptimizer};
#[allow(unused_imports)]
use std::io::{self, Write};
use std::time::Instant;

fn main() {
    // Take start time
    let start = Instant::now();

    // Define your objective function with required methods
    let objective_dim = 10;
    let objective_function = UserFitness::new(
        |individual: &nalgebra::DVector<f32>| individual.iter().map(|x| x.powi(2)).sum(),
        objective_dim,
        MinOrMax::Min,
    );
    let popsize = 15;

    println!("Trying to optimize a simple sum of squares function with CMA-ES (fold mode)...");
    println!("Number of dimensions: {}", objective_dim);
    println!("Population size: {}", popsize);

    // Initialize CMA-ES parameters
    let params = CmaesParams::new()
        .unwrap()
        .set_popsize(popsize)
        .unwrap()
        .set_xstart(objective_dim, 0.5)
        .unwrap()
        .set_sigma(0.5)
        .unwrap()
        .set_only_diag(true)
        .unwrap()
        // NOTE
        // If you set specific number of generations, you can fold easily through below
        .set_num_gens(50)
        .unwrap();

    // Create a new CMA-ES instance
    let cmaes = CmaesAlgo::new(params).unwrap();

    // Initialize the CMA-ES state
    let state = CmaesState::init_state(&cmaes.params).unwrap();

    // FOLD the CMA-ES algorithm until close to objective value
    let state = cmaes.rollout_fold(state, objective_function).unwrap();

    // Extract best solution and fitness
    let (best_y, best_y_fit) = state.get_best().unwrap();
    let elapsed = start.elapsed().as_secs_f32();
    let steps = cmaes.params.num_gens;
    let time_per_step = elapsed / steps as f32;

    // Print best candidate and fitness
    println!("\n========== OPTIMIZATION COMPLETE ==========");
    println!("Total Steps: {}", steps);
    println!("Total Time: {:.4} seconds", elapsed);
    println!("Time per Step: {:.6} seconds", time_per_step);
    println!("Best Fitness: {:+.3}", best_y_fit.row(0)[0]);
    println!("Best Individual: {:+.3}", best_y);
    println!("=======================================\n");
}
