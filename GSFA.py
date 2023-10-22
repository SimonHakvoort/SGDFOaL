import numpy as np
import matplotlib.pyplot as plt

from SGDFOaL.functions import Objective_J, GSFA

# Set SPSA hyperparameters
THETA_0 = np.array([5, -2])  # Initial guess for θ₁ and θ₂
EPSILON_TYPE = 'decreasing'  # Use 'fixed' or 'decreasing' for ε
EPSILON_VALUE = 0.01  # Initial value of ε if EPSILON_TYPE is 'fixed'
NR_ITERATIONS = 500  # Number of iterations
STOCHASTIC = False  # Set to True if you want to add stochasticity
MU = 0.0  # Mean of the stochastic noise
SIGMA = 0.1  # Standard deviation of the stochastic noise
BATCH = True  # Set to True for batch estimation of the gradient
NR_ESTIMATES = 10  # Number of estimates when BATCH is True
OPTIMIZATION_TYPE = 'minimization'  # 'minimization' or 'maximization'

grid_size = 0.1
nr_runs = 2

theta_1 = np.arange(-8, 8 + grid_size, grid_size)
theta_2 = np.arange(-8, 8 + grid_size, grid_size)
points = np.column_stack((theta_1.flatten(), theta_2.flatten()))

best_objective_value = np.inf
best_thetas = None
best_objective_values = None
best_gradients = None

for candidate in points:
    for i in range(nr_runs):
        # Run GSFA optimization
        thetas, gradients, objective_values = GSFA(Objective_J, candidate, EPSILON_TYPE, EPSILON_VALUE, NR_ITERATIONS,
                                                   STOCHASTIC, MU, SIGMA, BATCH, NR_ESTIMATES, OPTIMIZATION_TYPE)
        if objective_values[-1] < best_objective_value:
            best_objective_value = objective_values[-1]
            best_thetas = thetas
            best_objective_values = objective_values
            best_gradients = gradients
# Print the optimized thetas and objective value
print("Optimized thetas:", best_thetas[-1])
print("Optimized objective value:", best_objective_values[-1])

# Create a plot of thetas trajectory in [-8, 8]^2
plt.figure(figsize=(8, 8))
plt.plot(best_thetas[:, 0], best_thetas[:, 1], marker='o', linestyle='-', markersize=2, color='red', markerfacecolor='blue')
plt.xlim(-8, 8)
plt.ylim(-8, 8)
plt.xlabel('$\\theta_1$')
plt.ylabel('$\\theta_2$')
plt.title('Optimization Trajectory of $\\theta$ using GSFA')
plt.grid(True)

# Create a separate plot for the convergence of the objective value
plt.figure(figsize=(8, 5))
plt.plot(range(NR_ITERATIONS), best_objective_values)
plt.xlabel('Iteration')
plt.ylabel('Objective Value')
plt.title('Convergence of Objective Value using GSFA')
plt.grid(True)

# Display both figures with just one plt.show() at the end
plt.show()