use crate::fitness::{FitnessEvaluator, FitnessFunction};
use crate::params::{CmaesParams, CmaesParamsValidator};
use crate::state::{CmaesState, CmaesStateLogic};
use crate::strategy::{CmaesAlgo, CmaesAlgoOptimizer};
#[allow(unused_imports)]
use std::io::{self, Write};

pub fn express_executor(objective_function: impl FitnessFunction) -> impl CmaesStateLogic {
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
    state
}
