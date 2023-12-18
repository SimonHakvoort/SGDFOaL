import numpy as np
import matplotlib.pyplot as plt
from SGDFOaL.project_1 import RunExperiment

increment = 0.01

STOCHASTIC = False
MU = 0
SIGMA = 0.1
numRuns = 100000
objective_values = np.zeros((int(1 / increment) + 1, int(1 / increment) + 1))

p1_values = np.arange(0, 1 + increment, increment)
p2_values = np.arange(0, 1 + increment, increment)

for i in range(len(p1_values)):
    for j in range(len(p2_values)):
        p1 = p1_values[i]
        p2 = p2_values[j]
        if p1 + p2 <= 1:
            p3 = 1 - p1 - p2
            p = np.array([p1, p2, p3])
            sum = 0
            std = 0
            while std == 0:
                sum, std = RunExperiment(p, numRuns, STOCHASTIC, MU, SIGMA)
            objective_values[j,i] = np.mean(sum) / std


plt.figure()
levels = np.arange(0, 0.32 * 3 , 0.05)
contour = plt.contourf(p1_values, p2_values, objective_values, levels=levels, cmap='viridis')  # You can choose a different colormap


highlight_percentage = 3  # Change this percentage as needed
threshold_value = np.percentile(objective_values, 100 - highlight_percentage)
highlighted_values = np.ma.masked_where(objective_values <= threshold_value, objective_values)
plt.contourf(p1_values, p2_values, highlighted_values, levels=levels, colors='red', alpha=0.6)

# Add colorbar
cbar = plt.colorbar(contour)

# Add labels and title for the contour plot
plt.xlabel('$p_1$')
plt.ylabel('$p_2$')

plt.show()