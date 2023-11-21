import matplotlib.pyplot as plt
import numpy as np

from SGDFOaL.project_2 import RunExperiment

increment = 0.001
thetas = np.arange(0, 5 + increment, increment)
objective_values = np.zeros(len(thetas))

STOCHASTIC = False
MU = 0
SIGMA = 0.1
numRuns = 100000

for i in range(len(thetas)):
    quantile = RunExperiment(thetas[i], numRuns, STOCHASTIC, MU, SIGMA)
    objective_values[i] = np.mean((quantile - 8) ** 2)

plt.plot(thetas, objective_values)
plt.xlabel('$\\theta$')
plt.ylabel('Objective Value')
plt.xlim(0, 5)
plt.ylim(0, 80)
plt.show()