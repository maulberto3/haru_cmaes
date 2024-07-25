use criterion::{criterion_group, criterion_main, Criterion};
use haru_cmaes::simple_use;

// Benchmark functions
fn cmaes_benchmark(c: &mut Criterion) {
    c.bench_function("cmaes_benchmark", |b| {
        b.iter(|| simple_use::example().unwrap())
    });
}

// Main function to run benchmarks
criterion_group!(benches, cmaes_benchmark);

criterion_main!(benches);
