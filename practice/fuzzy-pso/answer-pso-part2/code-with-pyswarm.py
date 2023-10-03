import numpy as np
from pyswarm import pso

# Objective function
def objective_function(x):
    return (x[0] - 3.14) ** 2 + (x[1] - 2.72) ** 2 + np.sin(3 * x[0] + 1.41) + np.sin(4 * x[1] - 1.73)

# Set parameters
num_particles = 50
num_dimensions = 2
num_iterations = 100
c1 = 0.1
c2 = 0.1
w = 0.8
min_value = -10
max_value = 10

# Set bounds for the variables
lb = [min_value] * num_dimensions
ub = [max_value] * num_dimensions

# Perform PSO optimization
best_position, best_value = pso(objective_function, lb, ub, maxiter=num_iterations, swarmsize=num_particles, phip=c1, phig=c2, omega=w)

print("Optimal position:", best_position)
print("Optimal value:", best_value)
