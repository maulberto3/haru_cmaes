use criterion::{criterion_group, criterion_main, Criterion};
use haru_cmaes::fitness::{FitnessEvaluator, MinOrMax, UserFitness};
use haru_cmaes::params::{CmaesParams, CmaesParamsValidator};
use haru_cmaes::state::{CmaesState, CmaesStateLogic};
use haru_cmaes::strategy::{CmaesAlgo, CmaesAlgoOptimizer};
#[allow(unused_imports)]
use std::io::{self, Write};

fn ask_tell() {
    // Define a custom objective with a closure (sum of squares)
    let objective_dim = 10;
    let objective_function = UserFitness::new(
        |individual: &nalgebra::DVector<f32>| individual.iter().map(|x| x.powi(2)).sum(),
        objective_dim,
        MinOrMax::Min,
    );

    // Initialize CMA-ES parameters
    let popsize = 15;
    let params = CmaesParams::new()
        .unwrap()
        .set_popsize(popsize)
        .unwrap()
        .set_xstart(objective_function.evaluator_dim().unwrap(), 0.5)
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
        let mut fitness = objective_function.evaluate(&pop).unwrap();

        // Update the state with the new population and fitness values
        state = cmaes.tell(state, &mut pop, &mut fitness).unwrap();

        // Continue or done?
        if let Ok(true) = cmaes.is_done(&state, step) {
            break;
        }

        step += 1;
    }
}

fn cmaes_benchmark(c: &mut Criterion) {
    c.bench_function("CMA-ES Mine", |b| b.iter(|| ask_tell()));
}

criterion_group!(benches, cmaes_benchmark);
criterion_main!(benches);