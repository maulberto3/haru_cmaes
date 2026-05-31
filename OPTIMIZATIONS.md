
🔴 Top Bottlenecks:
Eigendecomposition (eigen_decomposition() in prepare_ask())

Heavy use of symmetric_eigen, symmetric_tridiagonal, householder
Called every generation - consider caching when covariance changes marginally
Use nalgebra BLAS feature: nalgebra = { features = ["openblas"] }
Matrix multiplications in tell() (rank-mu update)

The rank-mu fold loop with individual weights is creating many temporary matrices
Consider accumulating weights first, then one matrix operation
Frames show heavy array_axcpy, array_axc operations
Memory allocations

malloc, posix_memalign appear frequently
Pre-allocate reusable buffers instead of creating matrices each generation
Use copy_from() instead of cloning matrices
Matrix transposes in ask() and fitness evaluation

transpose_to_uninit appears many times
Consider storing transposed matrices or using views
Population sorting by fitness

Using stable_sort every generation - could use quicksort or partial_sort