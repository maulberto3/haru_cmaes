use anyhow::Result;
// include!("fold.rs");
use haru_cmaes::fitness::{FitnessEvaluator, MinOrMax, UserFitness};
use haru_cmaes::params::{CmaesParams, CmaesParamsValidator};
use haru_cmaes::state::{CmaesState, CmaesStateLogic};
use haru_cmaes::strategy::{CmaesAlgo, CmaesAlgoOptimizer};
use pprof::ProfilerGuard;
use std::fs::File;
// use nalgebra::DMatrix;
#[allow(unused_imports)]
use std::io::{self, Write};

fn profile_fold() {
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
    let _state = cmaes.rollout_fold(state, obj_func).unwrap();
}

fn main() -> Result<()> {
    // Start the profiler
    let guard = ProfilerGuard::new(1000)?;

    // Run the example
    for _i in 0..25 {
        profile_fold();
    }
    // println!("Finished simple CMA-ES optimization.");

    // Stop the profiler and generate the report
    if let Ok(report) = guard.report().build() {
        let mut file = File::create("examples/flamegraph.svg")?;
        report.flamegraph(&mut file)?;
    }

    println!("Flamegraph generated: examples/flamegraph.svg");
    Ok(())
}
