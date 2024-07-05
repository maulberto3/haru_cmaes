use anyhow::Result;

#[allow(unused_imports)]
// use blas_src;
mod fitness;
use fitness::square_and_sum;

mod params;
use params::CmaesParams;

mod state;
use state::CmaesState;

mod strategy;
use strategy::Cmaes;

/// `Result<T, Error>`
///
/// This is a reasonable return type to use throughout your application but also
/// for `fn main`; if you do, failures will be printed along with any
/// [context][Context] and a backtrace if one was captured.
///
/// `anyhow::Result` may be used with one *or* two type parameters.
///
pub fn work() -> Result<()> {
    // Simple Illustrative Algorithm Usage

    // Step 1: Choose initial parameters
    let params = CmaesParams {
        // Required
        // mean: vec![0.0],
        mean: vec![0.0, 1.0],
        // mean: vec![0.0; 20],
        sigma: 1.0,
        popsize: 6,
    };
    // dbg!(&params);

    // STEP 2: Instantiate Cmaes algorithm with parameters
    let cmaes = Cmaes::new(&params)?;
    // dbg!(&cmaes.params);

    // Step 3: Instantiate a Cmaes State
    let mut state = CmaesState::init_state(&params)?;
    // println!("{:+.4?}", &state);
    // println!("\n");

    // Step 4: Loop
    for i in 0..100 {
        let mut pop = cmaes.ask(&mut state)?;
        println!("\n");
        // dbg!(&pop.xs);
        let mut fitness = square_and_sum(&pop)?;
        // dbg!(&fitness.fit);
        state = cmaes.tell(state, &mut pop, &mut fitness)?;
        // println!("{:+.4?}", &state);
        println!("\n");
        dbg!(&state);
        // dbg!(&fitness);
    }
    // println!("{:+.4?}", &pop);
    // println!("\n");
    // let fitness = square_and_sum(&pop)?;
    // println!("{:+.4?}", &fit);

    // for _ in 0..100 {
    //     let pop: Array2<f32> = cmaes.ask(&state, &params);
    //     println!("{:+.4}", &pop);

    //     let fitness: Array2<f32> = square_and_sum(&pop);
    //     println!("{:+.4}", &fitness);

    //     state = cmaes.tell(pop, fitness, state, &params);
    //     println!("{:+.4?}", &state);
    //     println!("\n");
    // }
    // println!("{:+.4?}", &state);

    // let num_iters = 7;
    // for _i in 0..num_iters {
    // pop, state = cmaes.ask(state);
    // fit = fitness(&pop);
    // state = cmaes.tell(state, &pop, &fit, &params);
    // state.best_member, state.best_fitness

    //     break;
    // }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::work;

    #[test]
    // TODO: implement integration tests, similar to Robert Lange
    fn it_works() {
        _ = work();
    }
}
