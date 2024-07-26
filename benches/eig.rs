use anyhow::Result;
use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::DMatrix;
use ndarray::Array2;
use ndarray_linalg::Eig;
// use faer::prelude::*;
// use faer::mat;

/// Performs eigen decomposition using `ndarray-linalg`
///
/// # Arguments
///
/// * `matrix` - A reference to a 2D array of type `f32`
///
/// # Returns
///
/// A `Result` indicating success or failure.
fn eigen_ndarray(matrix: &Array2<f32>) -> Result<()> {
    let _ = matrix.clone().eig().unwrap();
    Ok(())
}

/// Performs eigen decomposition using `nalgebra`
///
/// # Arguments
///
/// * `matrix` - A reference to a `DMatrix` of type `f32`
///
/// # Returns
///
/// A `Result` indicating success or failure.
fn eigen_nalgebra(matrix: &DMatrix<f32>) -> Result<()> {
    let _ = matrix.clone().symmetric_eigen();
    Ok(())
}

/// Benchmark functions for eigen decomposition.
///
/// # Arguments
///
/// * `c` - A mutable reference to a `Criterion` object for benchmarking.
fn benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Eigen Decomposition");

    // Example matrix for testing
    let size = 100;
    let matrix_ndarray = Array2::<f32>::zeros((size, size));
    let matrix_nalgebra = DMatrix::<f32>::zeros(size, size);

    group.bench_function("Eigen Decomposition with ndarray-linalg", |b| {
        b.iter(|| eigen_ndarray(&matrix_ndarray))
    });
    group.bench_function("Eigen Decomposition with nalgebra", |b| {
        b.iter(|| eigen_nalgebra(&matrix_nalgebra))
    });
    group.finish();
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
