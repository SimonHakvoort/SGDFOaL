import numpy as np
import matplotlib.pyplot as plt
from SGDFOaL.project_1 import Objective

increment = 0.01
values = []

# Iterate over possible values for x1
for x1 in range(1, int(1/increment)):
    x1 *= increment

    # Iterate over possible values for x2
    for x2 in range(1, int((1 - x1)/increment)):
        x2 *= increment

        # Check the condition x1 + x2 < 1
        if x1 + x2 < 1:
            values.append((x1, x2, 1 - x1 - x2))

# Convert the list of tuples to a numpy array
values = np.array(values)


STOCHASTIC = False
MU = 0
SIGMA = 0.1
objective_values = []

# Iterate over all possible values for x1, x2 and x3
for x1, x2, x3 in values:
    objective_values.append(Objective(np.array([x1, x2, x3]), STOCHASTIC, MU, SIGMA))

plt.figure()
levels = np.arange(-500, 3200, 50)
contour = plt.contourf(values[0,:], values[1,:], objective_values, levels=levels, cmap='viridis')  # You can choose a different colormap

# Add a colorbar for reference
plt.colorbar(contour)

# Add labels and title for the contour plot
plt.xlabel('$\\theta_1$')
plt.ylabel('$\\theta_2$')
plt.title('2D Plot of a Function')

# Create the second figure for the 3D surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(values[0,:], values[1,:], objective_values, cmap='viridis')  # You can choose a different colormap

# Set labels and title for the 3D surface plot
ax.set_xlabel('$\\theta_1$')
ax.set_ylabel('$\\theta_2$')
ax.set_zlabel('$J$')
ax.set_title('3D Surface Plot of the Function')


# Display both plots
plt.show()