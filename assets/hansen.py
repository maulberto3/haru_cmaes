from cma import purecma as pcma

# Define the fitness function
def fitness_function(x):
    return sum(xi**2 for xi in x)

def example():
    # Initialize CMA-ES parameters
    dim = 50
    popsize = 50
    sigma = 0.75

    # Create a new CMA-ES instance
    es = pcma.CMAES(popsize * [0.0], sigma)
    
    # Run the CMA-ES algorithm for 150 iterations
    for _ in range(150):
        # Generate a new population
        X = es.ask()
        
        # Evaluate the fitness of the population
        f = [fitness_function(x) for x in X]
        
        # Update the state with the new population and fitness values
        es.tell(X, f)
    
    # Print the average fitness of the best solutions
    best_fitness = es.result[1]  # Best fitness value
    print(f"Fitness (best): {best_fitness:.4f}")

if __name__ == "__main__":
    example()
