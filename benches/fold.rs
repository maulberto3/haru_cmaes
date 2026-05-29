use criterion::{criterion_group, criterion_main, Criterion};
use haru_cmaes::fitness::{FitnessEvaluator, MinOrMax, UserFitness};
use haru_cmaes::params::{CmaesParams, CmaesParamsValidator};
use haru_cmaes::state::{CmaesState, CmaesStateLogic};
use haru_cmaes::strategy::{CmaesAlgo, CmaesAlgoOptimizer};
// use nalgebra::DMatrix;
use std::env::var;
#[allow(unused_imports)]
use std::io::{self, Write};
use std::time::Instant;

fn fold() {
    // Define verbose or not
    let verbose = var("VERBOSE").unwrap_or("No".to_string());

    // Take start time
    let start = Instant::now();

    // Define a custom objective with a closure (sum of squares)
    let objective_function = UserFitness::new(
        |individual: &nalgebra::DVector<f32>| individual.iter().map(|x| x.powi(2)).sum(),
        4,
        MinOrMax::Min,
    );

    // Initialize CMA-ES parameters
    let params = CmaesParams::new()
        .unwrap()
        .set_popsize(50)
        .unwrap()
        .set_xstart(objective_function.evaluator_dim().unwrap(), 0.5)
        .unwrap()
        .set_sigma(0.5)
        .unwrap()
        .set_only_diag(true)
        .unwrap()
        // NOTE
        // If you set specific number of generations, you can fold easily through below
        .set_num_gens(150)
        .unwrap();

    // Create a new CMA-ES instance
    let cmaes = CmaesAlgo::new(params).unwrap();

    // Initialize the CMA-ES state
    let state = CmaesState::init_state(&cmaes.params).unwrap();

    // FOLD the CMA-ES algorithm until close to objective value
    let state = cmaes.rollout_fold(state, objective_function).unwrap();

    // Print the average fitness of the best solutions
    if verbose != "No" {
        println!();
        println!(
            "Fitness: {:+.5?} | Duration p/step: {:.5} secs",
            &state.best_y_fit.row(0)[0],
            (start.elapsed().as_micros() as f32) / 1000000.0 / (cmaes.params.num_gens as f32)
        );
        // dbg!(state);
        println!("{:+.5?}", &state.best_y);
    }
}

fn cmaes_benchmark(c: &mut Criterion) {
    c.bench_function("CMA-ES FOLD", |b| b.iter(|| fold()));
}

criterion_group!(benches, cmaes_benchmark);
criterion_main!(benches);

// fn benchmarks(c: &mut Criterion) {
//     let mut group = c.benchmark_group("My Group");
//     group.bench_function("Function 1", |b| b.iter(|| function1()));
//     group.bench_function("Function 2", |b| b.iter(|| function2()));
//     group.finish();
// }
