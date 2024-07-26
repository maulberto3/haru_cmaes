use criterion::{criterion_group, criterion_main, Criterion};
use std::process::Command;

// To easily create a pythonenv to run hansen cma:
// ```
// python -m venv hansen_env
// source hansen_env/bin/activate
// pip install cma matplotlib
// ```

/// Run the Python script and capture its output.
///
/// # Arguments
///
/// * `script_path` - The path to the Python script to be executed.
/// * `env` - The path to the Python environment to use.
fn run_python_script(script_path: &str, env: &str) -> std::io::Result<String> {
    let output = Command::new("python")
        .arg(script_path)
        .env("PATH", env)
        .output()?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        Err(std::io::Error::new(std::io::ErrorKind::Other, "Python script failed"))
    }
}

/// Benchmark function for the Python CMA-ES script.
///
/// # Arguments
///
/// * `c` - A mutable reference to a `Criterion` object for benchmarking.
fn python_script_benchmark(c: &mut Criterion) {
    let python_script = "assets/hansen.py";
    let python_env = "../hansen_cma/bin"; // Adjust this to the path of your custom environment

    c.bench_function("CMA-ES Hansen", |b| {
        b.iter(|| {
            let _ = run_python_script(python_script, python_env).expect("Failed to run Python script");
        })
    });
}

criterion_group!(benches, python_script_benchmark);
criterion_main!(benches);
