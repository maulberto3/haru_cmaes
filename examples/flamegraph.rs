use anyhow::Result;
use haru_cmaes::ask_tell_use;
use pprof::ProfilerGuard;
use std::fs::File;

fn main() -> Result<()> {
    // Start the profiler
    let guard = ProfilerGuard::new(1000)?;

    // Run the example
    for _i in 0..200 {
        ask_tell_use::ask_tell_example()?;
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
