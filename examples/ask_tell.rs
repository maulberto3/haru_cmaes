use haru_cmaes::fitness::{FitnessEvaluator, MinOrMax, UserFitness};
use haru_cmaes::params::{CmaesParams, CmaesParamsValidator};
use haru_cmaes::state::{CmaesState, CmaesStateLogic};
use haru_cmaes::strategy::{CmaesAlgo, CmaesAlgoOptimizer};
// use nalgebra::DMatrix;
use std::env::var;
#[allow(unused_imports)]
use std::io::{self, Write};
use std::time::Instant;

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

    // Initialize CMA-ES parameters
    let params = CmaesParams::new()
        .unwrap()
        .set_popsize(50)
        .unwrap()
        .set_xstart(obj_func.evaluator_dim().unwrap(), 0.5)
        .unwrap()
        .set_sigma(0.5)
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
        let mut fitness = obj_func.evaluate(&pop).unwrap();

        // Update the state with the new population and fitness values
        state = cmaes.tell(state, &mut pop, &mut fitness).unwrap();

        // Continue or done?
        if let Ok(true) = cmaes.is_done(&state, step) {
            break;
        }

        // Log some info
        if verbose != "No" {
            print!("{:+.5?} ", &state.best_y_fit.row(0)[0]);
            // print!("best y {:+.5?} ", &state.best_y.row(0)[0]);
            io::stdout().flush().unwrap()
        }

        step += 1;
    }
    // Print the average fitness of the best solutions
    if verbose != "No" {
        println!();
        println!(
            "Step {} | Fitness: {:+.5?} | Duration p/step: {:.5} secs",
            step,
            &state.best_y_fit.row(0)[0],
            (start.elapsed().as_micros() as f32) / 1000000.0 / (step as f32)
        );
        // dbg!(state);
        println!("{:+.5?}", &state.best_y);
    }
}
