import numpy as np

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

# Initialize particles
particles = np.random.uniform(low=min_value, high=max_value, size=(num_particles, num_dimensions))
velocities = np.zeros((num_particles, num_dimensions))
best_positions = particles.copy()
best_values = np.zeros(num_particles)

# Perform PSO optimization
global_best_position = None
global_best_value = np.inf

for _ in range(num_iterations):
    for i in range(num_particles):
        current_position = particles[i]
        current_value = objective_function(current_position)

        if current_value < best_values[i]:
            best_positions[i] = current_position
            best_values[i] = current_value

        if current_value < global_best_value:
            global_best_position = current_position
            global_best_value = current_value

        r1 = np.random.random(num_dimensions)
        r2 = np.random.random(num_dimensions)

        cognitive_component = c1 * r1 * (best_positions[i] - current_position)
        social_component = c2 * r2 * (global_best_position - current_position)

        velocities[i] = w * velocities[i] + cognitive_component + social_component
        particles[i] = current_position + velocities[i]

print("Optimal position:", global_best_position)
print("Optimal value:", global_best_value)
