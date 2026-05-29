use haru_cmaes::fitness::MinOrMax;
use haru_cmaes::fitness::{FitnessEvaluator, UserFitness};
use haru_cmaes::params::{CmaesParams, CmaesParamsValidator};
use haru_cmaes::state::{CmaesState, CmaesStateLogic};
use haru_cmaes::strategy::{CmaesAlgo, CmaesAlgoOptimizer};
use std::env::var;
#[allow(unused_imports)]
use std::io::{self, Write};
use std::time::Instant;

fn express_executor(objective_function: impl FitnessEvaluator) -> (impl CmaesStateLogic, i32) {
    // Initialize CMA-ES parameters
    let params = CmaesParams::new()
        .unwrap()
        .set_popsize(objective_function.evaluator_dim().unwrap() as i32)
        .unwrap()
        .set_xstart(objective_function.evaluator_dim().unwrap(), 0.0)
        .unwrap()
        .set_only_diag(true)
        .unwrap();

    // Create a new CMA-ES instance
    let cmaes = CmaesAlgo::new(params).unwrap();

    // Initialize the CMA-ES state
    let mut state = CmaesState::init_state(&cmaes.params).unwrap();

    println!(
        "[CMA-ES] Starting optimization with objective dimension: {}",
        objective_function.evaluator_dim().unwrap()
    );

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
            println!("[CMA-ES] Convergence achieved at step {}", step);
            break;
        }

        step += 1;
    }
    (state, step)
}

fn main() {
    // Define verbose or not
    let verbose = var("VERBOSE").unwrap_or("No".to_string());

    // Take start time
    let start = Instant::now();

    // Define your objective function with required methods
    let obj_func = UserFitness::new(
        |individual: &nalgebra::DVector<f32>| individual.iter().map(|x| x.powi(2)).sum(),
        50,
        MinOrMax::Min,
    );

    println!("[EXPRESS] CMA-ES Optimization Example Started");
    println!("[EXPRESS] Objective: SquareAndSum (minimize sum of squares)");
    println!(
        "[EXPRESS] Objective Dimension: {}",
        obj_func.evaluator_dim().unwrap()
    );

    // Then, pass it to the simple executor:
    let (state, steps) = express_executor(obj_func);
    let (best_y, best_y_fit) = state.get_best().unwrap();

    let elapsed = start.elapsed().as_secs_f32();
    let time_per_step = elapsed / steps as f32;

    // Print best candidate and fitness
    println!("\n[EXPRESS] ========== OPTIMIZATION COMPLETE ==========");
    println!("[EXPRESS] Total Steps: {}", steps);
    println!("[EXPRESS] Total Time: {:.4} seconds", elapsed);
    println!("[EXPRESS] Time per Step: {:.6} seconds", time_per_step);
    println!("[EXPRESS] Best Fitness: {:+.8e}", best_y_fit.row(0)[0]);

    if verbose != "No" {
        println!("\n[EXPRESS] Best Solution Vector (first 10 components):");
        for (i, val) in best_y.row(0).iter().take(10).enumerate() {
            println!("[EXPRESS]   x[{}] = {:+.8e}", i, val);
        }
        if best_y.ncols() > 10 {
            println!("[EXPRESS]   ... ({} more components)", best_y.ncols() - 10);
        }
    }
    println!("[EXPRESS] =======================================\n");
}
