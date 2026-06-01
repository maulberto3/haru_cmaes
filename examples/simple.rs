use haru_cmaes::fitness::MinOrMax;
use haru_cmaes::fitness::{FitnessEvaluator, UserFitness};
use haru_cmaes::params::{CmaesParams, CmaesParamsValidator};
use haru_cmaes::state::{CmaesState, CmaesStateLogic};
use haru_cmaes::strategy::{CmaesAlgo, CmaesAlgoOptimizer};
use std::time::Instant;

fn express_executor(objective_function: impl FitnessEvaluator, popsize: i32) -> (impl CmaesStateLogic, i32) {
    // Initialize CMA-ES parameters
    let params = CmaesParams::new()
        .unwrap()
        .set_popsize(popsize)
        .unwrap()
        .set_xstart(objective_function.evaluator_dim().unwrap(), 0.0)
        .unwrap()
        .set_only_diag(true)
        .unwrap();

    // Create a new CMA-ES instance
    let cmaes = CmaesAlgo::new(params).unwrap();

    // Initialize the CMA-ES state
    let mut state = CmaesState::init_state(&cmaes.params).unwrap();

    // Run the CMA-ES algorithm until close to objective value
    let mut step = 1;
    loop {
        // Generate a new population
        let mut pop = cmaes.ask(&mut state).unwrap();

        // Evaluate the fitness of the population
        let mut fitness = objective_function.evaluate(&pop).unwrap();

        // Update the state with the new population and fitness values
        state = cmaes.tell(state, &mut pop, &mut fitness).unwrap();

        // Continue or done?
        if let Ok(true) = cmaes.is_done(&state, step) {
            break;
        }

        step += 1;
    }
    (state, step)
}

fn main() {
    // Take start time
    let start = Instant::now();

    // Define your objective function with required methods...
    let objective_dim = 10;
    let obj_func = UserFitness::new(
        |individual: &nalgebra::DVector<f32>| individual.iter().map(|x| x.powi(2)).sum(),
        objective_dim,
        MinOrMax::Min,
    );
    let popsize = 15;
    println!("Trying to optimize a simple sum of squares function with CMA-ES...");
    println!("Number of dimensions: {}", objective_dim);
    println!("Population size: {}", popsize);

    // ...then, pass it to the executor.
    let (state, steps) = express_executor(obj_func, popsize);
    let (best_y, best_y_fit) = state.get_best().unwrap();

    let elapsed = start.elapsed().as_secs_f32();
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
