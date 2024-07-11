// main.rs

use anyhow::Result;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::DMatrix;
use ndarray::Array2;
use ndarray_linalg::Eig;
use ndarray_rand::{rand_distr::StandardNormal, RandomExt};

// Define eigen decomposition using ndarray-linalg
fn eigen_decomp_ndarray_linalg(matrix: &Array2<f64>) -> Result<()> {
    let _res = matrix.eig().unwrap();
    Ok(())
}

// Define eigen decomposition using nalgebra
fn eigen_decomp_nalgebra(matrix: &DMatrix<f64>) -> Result<()> {
    let _res = matrix.symmetric_eigenvalues();
    Ok(())
}

// Benchmark functions
fn eigen_decomp_ndarray_linalg_benchmark(c: &mut Criterion) {
    let matrix = Array2::<f64>::random((50, 50), StandardNormal); // Example: random 50x50 matrix
    c.bench_function("eigen_decomp_ndarray_linalg", |b| {
        b.iter(|| eigen_decomp_ndarray_linalg(black_box(&matrix)))
    });
}

fn eigen_decomp_nalgebra_benchmark(c: &mut Criterion) {
    let matrix = DMatrix::<f64>::from_row_slice(
        50,
        50,
        &Array2::<f64>::random((50, 50), StandardNormal).into_raw_vec(),
    ); // Example: random 50x50 matrix
    c.bench_function("eigen_decomp_nalgebra", |b| {
        b.iter(|| eigen_decomp_nalgebra(black_box(&matrix)))
    });
}

// Main function to run benchmarks
criterion_group!(
    benches,
    eigen_decomp_ndarray_linalg_benchmark,
    eigen_decomp_nalgebra_benchmark
);

criterion_main!(benches);
