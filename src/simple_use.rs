use crate::{fitness::square_and_sum, params::CmaesParams, state::CmaesState, strategy::Cmaes};
use anyhow::Result;

#[allow(dead_code)]
pub fn example() -> Result<()> {
    // Simple Illustrative Algorithm Usage
    let params = CmaesParams {
        popsize: 50,
        xstart: vec![0.0; 50],
        sigma: 0.75,
    };

    let cmaes = Cmaes::new(&params)?;
    let mut state = CmaesState::init_state(&params)?;
    for _i in 0..150 {
        let mut pop = cmaes.ask(&mut state)?;
        let mut fitness = square_and_sum(&pop)?;
        // println!(
        //     "{:+.4?} {:+.4?}",
        //     &fitness.values.mean(),
        //     &state.best_y_fit.mean(),
        // );
        state = cmaes.tell(state, &mut pop, &mut fitness)?;
    }

    // println!("Best y: {:+.4?}", &state.best_y);
    // println!("Best y (fitness): {:+.4?}", &state.best_y_fit);
    println!("Fitness (mean): {:+.4?}", &state.best_y_fit.mean());

    Ok(())
}
