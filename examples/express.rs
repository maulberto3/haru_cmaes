use haru_cmaes::fitness::MinOrMax;
use haru_cmaes::fitness::{FitnessEvaluator, FitnessFunction};
use haru_cmaes::objectives::SquareAndSum;
use haru_cmaes::params::{CmaesParams, CmaesParamsValidator};
use haru_cmaes::state::{CmaesState, CmaesStateLogic};
use haru_cmaes::strategy::{CmaesAlgo, CmaesAlgoOptimizer};
use std::env::var;
#[allow(unused_imports)]
use std::io::{self, Write};
use std::time::Instant;

fn express_executor(objective_function: impl FitnessFunction) -> (impl CmaesStateLogic, i32) {
    // Initialize CMA-ES parameters
    let params = CmaesParams::new()
        .unwrap()
        .set_popsize(objective_function.cost_dim() as i32)
        .unwrap()
        .set_xstart(objective_function.cost_dim(), 0.0)
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
    // Define verbose or not
    let verbose = var("VERBOSE").unwrap_or("No".to_string());

    // Take start time
    let start = Instant::now();

    // Define your objective function with required methods
    let obj_func = SquareAndSum {
        obj_dim: 50,
        dir: MinOrMax::Min,
    };

    // Then, pass it to the simple executor:
    let (state, steps) = express_executor(obj_func);
    let (best_y, best_y_fit) = state.get_best().unwrap();

    // Print best candidate and fitness
    if verbose != "No" {
        println!();
        println!(
            "Fitness: {:+.5?} | Duration p/step: {:.5} secs",
            best_y_fit.row(0)[0],
            (start.elapsed().as_micros() as f32) / 1000000.0 / (steps as f32)
        );
        // dbg!(state);
        println!("{:+.5?}", best_y);
    }
}
