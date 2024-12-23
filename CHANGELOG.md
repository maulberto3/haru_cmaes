# Changelog

All notable changes to this project will be documented in this file.

## [unreleased]

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
