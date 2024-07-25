use anyhow::Result;
use ndarray::{Array2, Axis};

use crate::strategy::PopulationY;

#[derive(Debug, Clone)]
pub struct Fitness {
    pub values: Array2<f32>,
}

pub fn square_and_sum(pop: &PopulationY) -> Result<Fitness> {
    let values = pop
        .y
        .map_axis(Axis(1), |row| row.mapv(|elem| elem.powi(2)).sum())
        .view()
        .into_shape((pop.y.nrows(), 1))
        .unwrap()
        .to_owned();
    Ok(Fitness { values })
}

pub fn simple_std(pop: &PopulationY) -> Result<Fitness> {
    let values = pop
        .y
        .map_axis(Axis(1), |row| row.std(1.0))
        .view()
        .into_shape((pop.y.nrows(), 1))
        .unwrap()
        .to_owned();
    Ok(Fitness { values })
}
// TODO:
// Implement other objective functions
// simple std
// DEA would be great
// Rastrigin and others
