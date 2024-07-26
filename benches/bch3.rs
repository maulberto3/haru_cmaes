use anyhow::Result;
use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::DMatrix;
use ndarray::Array2;
use ndarray_linalg::Eig;
// use faer::prelude::*;
// use faer::mat;

fn eigen_ndarray(matrix: &Array2<f64>) -> Result<()> {
    let _ = matrix.clone().eig().unwrap();
    Ok(())
}

fn eigen_nalgebra(matrix: &DMatrix<f64>) -> Result<()> {
    let _ = matrix.clone().symmetric_eigen();
    Ok(())
}

// fn eigen_faer(matrix: &Mat<f64>) -> Result<(), ()> {
//     let _ = matrix.eigendecomposition::<c64>();
//     Ok(())
// }

fn benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Eigen Decomposition");

    // Example matrix for testing
    let size = 100;
    let matrix_ndarray = Array2::<f64>::zeros((size, size));
    let matrix_nalgebra = DMatrix::<f64>::zeros(size, size);
    // let matrix_faer = Mat::zeros(size, size);

    group.bench_function("Eigen Decomposition with ndarray-linalg", |b| {
        b.iter(|| eigen_ndarray(&matrix_ndarray))
    });
    group.bench_function("Eigen Decomposition with nalgebra", |b| {
        b.iter(|| eigen_nalgebra(&matrix_nalgebra))
    });
    // group.bench_function("Eigen Decomposition with faer", |b| {
    //     b.iter(|| eigen_faer(&matrix_faer).unwrap())
    // });

    group.finish();
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
