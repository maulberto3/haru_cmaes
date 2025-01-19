use haru_cmaes::express_use::express_executor;
use haru_cmaes::fitness::MinOrMax;
use haru_cmaes::objectives::SquareAndSum;
use haru_cmaes::state::CmaesStateLogic;

// Example usage of the express executor:
// CMA-ES algorithm with default configuration
fn main() {
    // First, defines your objective function
    // with required methods
    let obj_func = SquareAndSum {
        obj_dim: 50,
        dir: MinOrMax::Min,
    };

    // Then, pass it to the simple executor,
    let state = express_executor(obj_func);
    let (best_y, best_y_fit) = state.get_best().unwrap();

    // Print best candidate and fitness
    println!("{}", best_y_fit);
    println!("{}", best_y);
}
