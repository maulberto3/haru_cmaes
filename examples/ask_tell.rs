use haru_cmaes::ask_tell_use;

fn main() {
    // Example usage of the CMA-ES algorithm.
    //
    // First, it defines and checks that your objective
    // function is allowed
    //
    // Then, create default Cmaes parameters, after which
    // can be adjusted to specific values
    //
    // Then, we pass those params to Cmaes algorithm and start
    // the ask-tell loop
    //
    // Here, it is shown that if the mean of last 25 best
    // optimization values is close to current best,
    // the lops breaks
    //
    // Finally, the solution is under state.best_y and state.best_y_fit
    #[allow(unused_variables)]
    let state = ask_tell_use::ask_tell_example().unwrap();
}
