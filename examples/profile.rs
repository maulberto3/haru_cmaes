use haru_cmaes::simple_use;
use pprof::ProfilerGuard;
use std::fs::File;
use anyhow::Result;

fn main() -> Result<()> {
    // Start the profiler
    let guard = ProfilerGuard::new(1000)?;

    // Run the example
    for _i in 0..100 {
        simple_use::example()?;
    }
    println!("Finished simple CMA-ES optimization.");

    // Stop the profiler and generate the report
    if let Ok(report) = guard.report().build() {
        let mut file = File::create("examples/flamegraph.svg")?;
        report.flamegraph(&mut file)?;
    }

    println!("Flamegraph generated: examples/flamegraph.svg");
    Ok(())
}
