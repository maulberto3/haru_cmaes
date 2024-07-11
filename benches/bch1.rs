// benches/eigen_benchmark.rs

use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{array, Array2};
use ndarray_linalg::Eig;

pub fn eigen_decomposition_benchmark(c: &mut Criterion) {
    let a: Array2<f64> = array![[1.01, 0.86, 4.60], [3.98, -0.53, 7.04], [3.30, 8.26, 3.89],];

    c.bench_function("eigen_decomposition", |b| {
        b.iter(|| {
            let (_eigs, _vecs) = a.clone().eig().unwrap();
            // // Ensure the calculation produces a result that is used
            // black_box(eigs);
            // black_box(vecs);
        });
    });
}

criterion_group!(benches, eigen_decomposition_benchmark);
criterion_main!(benches);
