import numpy as np
import matplotlib.pyplot as plt

from functions import Objective_J

STOCHASTIC = False
MU = 0
SIGMA = 1

# Generate values for theta within the specified range
theta = np.linspace(-8, 8, 1000)  # 1000 points between -8 and 8

# Create a grid of (theta, theta) values using meshgrid
Theta1, Theta2 = np.meshgrid(theta, theta)

# Combine Theta1 and Theta2 into a 2D array
theta_grid = np.array([Theta1, Theta2])

# Compute the function values for each (theta, theta) pair
Z = Objective_J(theta_grid, STOCHASTIC, MU, SIGMA)

# Create the first figure for the contour plot
plt.figure()
levels = np.arange(-500, 3200, 50)
contour = plt.contourf(Theta1, Theta2, Z, levels=levels, cmap='viridis')  # You can choose a different colormap

# Add a colorbar for reference
plt.colorbar(contour)

# Add labels and title for the contour plot
plt.xlabel('$\\theta_1$')
plt.ylabel('$\\theta_2$')
plt.title('2D Plot of a Function')

# Create the second figure for the 3D surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(Theta1, Theta2, Z, cmap='viridis')  # You can choose a different colormap

# Set labels and title for the 3D surface plot
ax.set_xlabel('$\\theta_1$')
ax.set_ylabel('$\\theta_2$')
ax.set_zlabel('$J$')
ax.set_title('3D Surface Plot of the Function')


# Display both plots
plt.show()