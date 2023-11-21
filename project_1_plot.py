import numpy as np
import matplotlib.pyplot as plt
from SGDFOaL.project_1 import RunExperiment

increment = 0.01
values = []

STOCHASTIC = False
MU = 0
SIGMA = 0.1
numRuns = 100000
objective_values = np.zeros((int(1 / increment) + 1, int(1 / increment) + 1))

p1_values = np.arange(0, 1 + increment, increment)
p3_values = np.arange(0, 1 + increment, increment)

for i in range(len(p1_values)):
    for j in range(len(p3_values)):
        p1 = p1_values[i]
        p3 = p3_values[j]
        if p1 + p3 <= 1:
            p2 = 1 - p1 - p3
            p = np.array([p1, p2, p3])
            sum = 0
            std = 0
            while std == 0:
                sum, std = RunExperiment(p, numRuns, STOCHASTIC, MU, SIGMA)
            objective_values[i, j] = np.mean(sum) / std


plt.figure()
levels = np.arange(0, 0.32, 0.01)
contour = plt.contourf(p1_values, p3_values, objective_values, levels=levels, cmap='viridis')  # You can choose a different colormap

# Add a colorbar for reference
plt.colorbar(contour)

# Add labels and title for the contour plot
plt.xlabel('$p_1$')
plt.ylabel('$p_3$')

plt.show()