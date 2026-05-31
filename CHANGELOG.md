# Changelog

All notable changes to this project will be documented in this file.

## [unreleased]

### 🚀 Features

- Enhance logging for CMA-ES optimization process and results
- Replace SquareAndSum with UserFitness closure for custom objective in benchmarks
- Replace SquareAndSum with UserFitness closure for objective functions in examples
- Implement UserFitness closure for custom fitness evaluation in PopulationY

### 🐛 Bug Fixes

- Add dev branch to push trigger in Rust setup workflow
- Update echo quotes in cache hit messages for consistency
- Update cache hit messages to use notice format for consistency
- Remove notice format from cache hit messages for consistency
- Remove unnecessary flags from cargo build and run commands for simplicity

### 🚜 Refactor

- Enhance Rust setup workflow with verbose build and test outputs
- Comment out unit tests in Rust setup workflow
- Add cache hit messages for APT packages, Cargo registry, and Rust toolchain
- Remove legacy objective functions and related implementations
- Replace FitnessFunction with FitnessEvaluator in CmaesAlgoOptimizer
- Use is_multiple_of for clarity in median function
- Update .gitignore and remove objectives module from lib.rs
- Streamline Makefile commands for improved clarity and efficiency

## [1.0.9] - 2025-02-03

### 🚜 Refactor

- All examples no bin, only in folder

## [1.0.8] - 2025-02-03

### 🚀 Features

- Fold interface for specific number of generations

### 🚜 Refactor

- Contain profile_memory feature in cfg

## [1.0.7] - 2025-01-27

### 🚀 Features

- Faster rand
- Faster rand
- Samply (similar to flamegraph)
- Default only diag; more clarity

### 🐛 Bug Fixes

- Stop tracking files
- Stop tracking files
- Small clean ups
- Temporal fix of make test cmaes yeslogic-fontconfig-sys issue
- Cleanup
- Minor cleanups (revert perf-event2)
- Remove sign under XSquare objective function
- Enforcing (random) sparsity in cov introduced singularity
- Minor cleanup
- Clippy warnings
- Clippy warnings
- More clarity in first test
- Normal draw had a bug

### 🚜 Refactor

- Updated github actions

### ⚙️ Miscellaneous Tasks

- Cargo test

### ◀️ Revert

- Bring back bins

## [1.0.6] - 2025-01-19

### 🚀 Features

- New simple express executor

### 🐛 Bug Fixes

- Removed function
- Remove argmin bench
- Removed unsued f_major and nlopt
- Url proper format for doc

### 🚜 Refactor

- Added with_capacity for some vecs; needed update trait
- Rename
- Rename for clarity; simpler vec creation;

### Build

- New artifacts

## [1.0.5] - 2025-01-05

### 🐛 Bug Fixes

- To account only for when others dea passes 1.0
- Removed now obsolete from ndarray (allowed unused) import of backend
- Minor cleanups
- Bump version
- Removed custom error

### 🚜 Refactor

- Constraint and DEA functions
- More clarity on blanket impl
- Build distr once, then call it
- With iter instead of for
- Fold for clarity on cov aggregation, plus no degradation

### 📚 Documentation

- More clarity on why specify a bin entry point

### 🧪 Testing

- Added doctests and minor redundancy
- Added doctest

### Build

- Some build artifacts

## [1.0.4] - 2025-01-02

### 🐛 Bug Fixes

- Minor cleanup
- Bump version after fulll build

### Build

- New version artifacts and cleanup

## [1.0.2] - 2025-01-02

### 🚀 Features

- Custom median function for evaluation
- Allow historical tracking of best candidates fitness
- Allow for min or max objective funtions
- More optimization precision by default
- Added required # of steps before first output
- Latest example
- Added ConstraintProblem and DEAProblem examples

### 🐛 Bug Fixes

- Test is now obsolete due to nalgebra crate adoption
- *(covariance)* Enforce sparsity for non-diagonal elements and positivity for diagonal elements

### 🚜 Refactor

- Decouple fitness with objectives
- Improved clarity for associated type
- A bit of cleanup for clarity
- Latest example
- Minor cleanup

### ⚡ Performance

- More samples for cpu profiler
- Added blanket impl for fitness functions

### 🧪 Testing

- Added tests
- Improved tests for clarity
- Added test for state (plus a bit of cleanup)
- Added tests

### ⚙️ Miscellaneous Tasks

- Minor clean up

## [1.0.1] - 2024-12-30

### 🐛 Bug Fixes

- Bump version

### 🚜 Refactor

- Better looking graph-dep png

### ⚙️ Miscellaneous Tasks

- Some ci artifacts and fixed minor issues

## [1.0.0] - 2024-12-30

### 🚜 Refactor

- Full migration to nalgebra

## [0.6.10] - 2024-12-28

### 🚀 Features

- Openblas temporal link

### 🐛 Bug Fixes

- Temporal remove of feature flag as it is now default
- Bump version, already taken by git tag

### 🧪 Testing

- Doc tests for tests and integration tests for tarpaulin

### Build

- Cleanup
- Tarpaulin new report

## [0.6.9] - 2024-12-28

### 🚀 Features

- *(strategy)* Fix closeness to target stopping criteria
- Added Rastrigin function
- Continue to improve user interface

### 🐛 Bug Fixes

- *(params)* Remove obj_value param, update all params if setter
- *(README)* For example now refer to simple_use.rs
- Minor type redundacy
- Bump version

### 🚜 Refactor

- *(fitness)* Define fitness and then apply trait
- *(strategy)* A few refactos
- *(state)* A few refactors
- Update explanation and verbose flag

### 📚 Documentation

- *(fitness)* More context on the doctest for SquareAndSum
- *(fitness)* More context on the doctest for SquareAndSum

### ⚡ Performance

- *(strategy)* Less verbose methods
- Added memory profile (linux) conditional cfg
- Unlink backend to end users (as suggested by devs)

### Build

- Specific ndarray version
- Full build outputs

## [0.6.8] - 2024-12-25

### 🐛 Bug Fixes

- Use blas_src already at lib.rs

### 🧪 Testing

- *(fitness)* Test of example square and sum
- New coverage
- *(fitness)* Test of example square and sum
- New coverage

## [0.6.7] - 2024-12-25

### 🚜 Refactor

- Easier allow users to provide cost function

### 🧪 Testing

- New coverage

## [0.6.6] - 2024-12-25

### 🚀 Features

- Setters for fundamental cmaes parameters

### 🐛 Bug Fixes

- Remove git push in make publ, it is already in make clif
- Remove make tree from make

### 🎨 Styling

- Fix spaces by cargo fmt

### 🧪 Testing

- *(params)* Doctests
- New coverage html

## [0.6.5] - 2024-12-25

### 🚜 Refactor

- Refactoring params for builder pattern

## [0.6.4b] - 2024-12-23

### 🐛 Bug Fixes

- Since it is a mod, it needs it through Cargo.toml
- Git tag workaround

## [0.6.4] - 2024-12-23

### 🚀 Features

- *(CmaesState)* Performance enhancement due to sparse covariance efforts
- Allow covariance sparsity as parameter
- Re-exports for easy use
- *(fitness)* Use of associated types in traits

### 🐛 Bug Fixes

- Inequality typo

### 🚜 Refactor

- *(CmaesAlgoOptimizer)* Better trait redability and use of associated type

### 📚 Documentation

- Started doing doc tests with params
- Updated README

### ⚙️ Miscellaneous Tasks

- Some build artifacts

## [0.6.3] - 2024-12-10

### 📚 Documentation

- Bump version

## [0.6.2] - 2024-12-10

### 🚜 Refactor

- Use match instead of if statements
- Better readability for CmaesAlgo
- Better readability for params
- Make use of associate type in trait
- Better git cliff use
- Makefile with git cliff
- Makefile
- Makefile

### 📚 Documentation

- New todo builder pattern
- Changelog

### Build

- Minor artifacts from build

## [0.6.1] - 2024-12-03

### 🚀 Features

- Added git-cliff and coverage

### 🐛 Bug Fixes

- Removed nlopt as it depends on C
- Disable into_f_major method

### 🚜 Refactor

- Makefile
- Minor changes from build

### 📚 Documentation

- Minor comments
- Pending TODO
- Experimenting with specific build flags

### 🎨 Styling

- Minor whitespaces
- Indent params

### Build

- Cargo build

<!-- generated by git-cliff -->
