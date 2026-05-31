use haru_cmaes::fitness::{FitnessEvaluator, MinOrMax, UserFitness};
use haru_cmaes::params::{CmaesParams, CmaesParamsValidator};
use haru_cmaes::state::{CmaesState, CmaesStateLogic};
use haru_cmaes::strategy::{CmaesAlgo, CmaesAlgoOptimizer};

/// Objective function dimension.
///
/// **CRITICAL:** This must match:
/// - UserFitness objective dimension parameter
/// - CmaesParams xstart dimension
/// - Closure implementation (number of dimensions accessed)
///
/// Mismatch will cause panics during optimization.
const OBJECTIVE_DIM: usize = 10;

/// Population size (can differ from objective dimension).
const POPSIZE: i32 = 10;

#[test]
fn test_express_optimization() {
    // ========== Set random seed for reproducibility ==========
    fastrand::seed(42);

    // ========== Define the objective function ==========
    // Create a UserFitness with a closure that computes sum of squares.
    // The dimension parameter (OBJECTIVE_DIM) must match the closure implementation.
    let obj_func = UserFitness::new(
        |individual: &nalgebra::DVector<f32>| individual.iter().map(|x| x.powi(2)).sum(),
        OBJECTIVE_DIM,
        MinOrMax::Min,
    );

    // ========== Configure CMA-ES parameters ==========
    // Set up the optimization with:
    // - Population size (can differ from objective dimension)
    // - Initial solution starting at origin
    // - Diagonal-only covariance matrix for efficiency
    let params = CmaesParams::new()
        .unwrap()
        .set_popsize(POPSIZE)
        .unwrap()
        .set_xstart(OBJECTIVE_DIM, 0.0)
        .unwrap()
        .set_tol(0.001)
        .unwrap()
        .set_only_diag(false)
        .unwrap();

    // ========== Initialize CMA-ES optimizer ==========
    let cmaes = CmaesAlgo::new(params).unwrap();
    let mut state = CmaesState::init_state(&cmaes.params).unwrap();

    // ========== Run optimization loop ==========
    // Continue until convergence is detected by is_done().
    let mut step = 1;
    loop {
        // Generate candidate population
        let mut pop = cmaes.ask(&mut state).unwrap();

        // Evaluate fitness of each candidate
        let mut fitness = obj_func.evaluate(&pop).unwrap();

        // Update CMA-ES state with results and get new distribution parameters
        state = cmaes.tell(state, &mut pop, &mut fitness).unwrap();

        // Check if optimization has converged i.e. uses cmaes tolerance parameter to determine if best fitness is close enough to optimum.
        if let Ok(true) = cmaes.is_done(&state, step) {
            break;
        }
        step += 1;
    }

    // ========== Verify convergence to optimum ==========
    // Extract the best solution found during optimization.
    let (best_solution, best_fitness) = state.get_best().unwrap();

    // For the objective function f(x) = sum(x_i^2):
    // - Global optimum is at x* = [0, 0, ..., 0]^T
    // - Optimal fitness value is f(x*) = 0
    //
    // Assert that we have converged within tolerance of the global optimum.
    // Tolerance of 0.1 indicates convergence for a 10-dimensional minimization problem.
    let fitness_value = best_fitness.row(0)[0];
    assert!(
        fitness_value < 0.1,
        "Optimization failed: best fitness {} exceeds tolerance of 0.1. \
         Expected convergence to global optimum near f(x*) = 0.",
        fitness_value
    );

    // Display results in human-readable format
    eprintln!("\n========== OPTIMIZATION RESULTS ==========");
    eprintln!("Best fitness value: {:.3}", fitness_value);
    eprintln!("Number of steps to convergence: {}", step);
    eprintln!("\nBest coefficients found:");
    for (i, val) in best_solution.column(0).iter().enumerate() {
        eprintln!("  x[{:2}] = {:8.3}", i, val);
    }
    eprintln!("========================================\n");
}
