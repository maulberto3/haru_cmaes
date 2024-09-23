use criterion::{criterion_group, criterion_main, Criterion};
include!("../assets/argmin.rs");

/// Benchmark function for the `cmaes` example.
///
/// This function measures the performance of the `example` function from the `simple_use` module.
///
/// # Arguments
///
/// * `c` - A mutable reference to a `Criterion` object for benchmarking.
fn cmaes_benchmark(c: &mut Criterion) {
    c.bench_function("Argmin PSO", |b| b.iter(|| example()));
}

criterion_group!(benches, cmaes_benchmark);
criterion_main!(benches);

// fn benchmarks(c: &mut Criterion) {
//     let mut group = c.benchmark_group("My Group");
//     group.bench_function("Function 1", |b| b.iter(|| function1()));
//     group.bench_function("Function 2", |b| b.iter(|| function2()));
//     group.finish();
// }
