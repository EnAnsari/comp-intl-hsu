import numpy as np
from scipy.optimize import minimize

# Objective function
def objective_function(x):
    return (x[0] - 3.14) ** 2 + (x[1] - 2.72) ** 2 + np.sin(3 * x[0] + 1.41) + np.sin(4 * x[1] - 1.73)

# Initial guess for optimization
initial_guess = [0, 0]

# Perform optimization
result = minimize(objective_function, initial_guess, method='BFGS')
optimal_position = result.x
optimal_value = result.fun

print("Optimal position (SciPy):", optimal_position)
print("Optimal value (SciPy):", optimal_value)
