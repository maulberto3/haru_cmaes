use anyhow::Result;
// include!("fold.rs");
use haru_cmaes::fitness::{FitnessEvaluator, MinOrMax};
use haru_cmaes::objectives::SquareAndSum;
use haru_cmaes::params::{CmaesParams, CmaesParamsValidator};
use haru_cmaes::state::{CmaesState, CmaesStateLogic};
use haru_cmaes::strategy::{CmaesAlgo, CmaesAlgoOptimizer};
use pprof::ProfilerGuard;
use std::fs::File;
// use nalgebra::DMatrix;
#[allow(unused_imports)]
use std::io::{self, Write};

fn profile_fold() {
    // Create cost function, i.e. see fitness.rs for an example
    let obj_func = SquareAndSum {
        obj_dim: 50,
        dir: MinOrMax::Min,
        // target: 1.0,
        // output_dim: 2,
        // input_dim: 2,
        // data: DMatrix::from_row_slice(3, 4, &vec![
        //     1.2, 3.0, 2.0, 2.5,
        //     4.0, 5.5, 6.0, 2.5,
        //     6.0, 8.5, 7.0, 9.5,
        // ])
    };

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
