# CMAES in Rust

## Motivation

This is my own Rust implementation of the CMA-ES optimization algorithm based in Hansen's purecma python implementation.

## Roadmap

Now that I have migrated this crate to nalgebra matrix crate, current roadmap is to refactor all code for maximum performance.

Also, I plan to provide some interface utilities for users and provide more examples on how to use this tool.

## Usage examples

Please see:
    - ask_tell_use.rs for a detailed and flexible use,
    - express_use.rs for a quick run with default configurations,
    - fold_use.rs to run for a specific number of generations.
        - To run other fitness examples, please see objectives.rs and adjust accordingly i.e. commented out code

## About Backend

Although, I have conditionally configured-coded this tool to make use of openblas, netlib, accelerate and/or intel-mkl backends, I have noit yet tested them. 

### How to contribute?

You can contribute any way you like.

Current suggestions:
    - Test backends functionality
    - More documentation
    - Bugs
    - Any other thing you'd like to contribute