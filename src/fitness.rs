use anyhow::Result;
use ndarray::{Array1, Axis};

use crate::strategy::Population;

#[derive(Debug, Clone)]
pub struct Fitness {
    pub fit: Array1<f32>,
}

pub fn square_and_sum(pop: &Population) -> Result<Fitness> {
    // let popsize = pop.shape()[0];
    let fit = pop
        .xs
        .map_axis(Axis(1), |row| row.mapv(|elem| elem.powi(2)).sum());
    Ok(Fitness { fit })
}

// TODO:
// Implement other objective functions
// simple std
// DEA would be great
// Rastrigin and others
