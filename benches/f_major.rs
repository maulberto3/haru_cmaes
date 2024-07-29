use anyhow::Result;
use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::Array2;
use ndarray_linalg::Eig;

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

/// Creates a matrix in row-major order and column-major order
///
/// # Arguments
///
/// * `size` - The size of the matrix
///
/// # Returns
///
/// A tuple containing the row-major and column-major matrices.
fn create_matrices(size: usize) -> (Array2<f32>, Array2<f32>) {
    let row_major = Array2::<f32>::zeros((size, size));
    
    // Convert row-major to column-major
    let column_major = row_major.t().to_owned();

    (row_major, column_major)
}

/// Benchmark functions for eigen decomposition.
///
/// # Arguments
///
/// * `c` - A mutable reference to a `Criterion` object for benchmarking.
fn benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Eigen Decomposition");

    // Example matrix for testing
    let size = 1000;
    let (matrix_row_major, matrix_column_major) = create_matrices(size);

    group.bench_function("Eigen Decomposition with row-major ndarray-linalg", |b| {
        b.iter(|| eigen_ndarray(&matrix_row_major))
    });

    group.bench_function("Eigen Decomposition with column-major ndarray-linalg", |b| {
        b.iter(|| eigen_ndarray(&matrix_column_major))
    });

    group.finish();
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);


// use criterion::{criterion_group, criterion_main, Criterion};
// use ndarray::{Array2, ArrayView2};

// // Matrix multiplication with row-major storage
// fn matmul_row_major(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
//     a.to_owned().dot(b)
// }

// // Matrix multiplication with column-major storage
// fn matmul_col_major(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
//     a.to_owned().dot(b)
// }

// fn benchmarks(c: &mut Criterion) {
//     let mut group = c.benchmark_group("Matrix Multiplication");

//     // Example matrix for testing
//     let size = 100;
//     let matrix_a: Array2<f32> = Array2::<f32>::zeros((size, size));
//     let matrix_b: Array2<f32> = Array2::<f32>::zeros((size, size));
//     let matrix_a_col_major: Array2<f32> = Array2::<f32>::zeros((size, size)).t().to_owned();
//     let matrix_b_col_major: Array2<f32> = Array2::<f32>::zeros((size, size)).t().to_owned();

//     group.bench_function("Matrix Multiplication with ndarray-linalg (row-major)", |b| {
//         b.iter(|| matmul_row_major(&matrix_a, &matrix_b))
//     });

//     group.bench_function("Matrix Multiplication with ndarray-linalg (column-major)", |b| {
//         b.iter(|| matmul_col_major(&matrix_a_col_major, &matrix_b_col_major))
//     });

//     group.finish();
// }

// criterion_group!(benches, benchmarks);
// criterion_main!(benches);
