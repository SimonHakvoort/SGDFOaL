import numpy as np
from matplotlib import pyplot as plt

from SGDFOaL.functions import GradientDescent, estimate_gradient_spsa, estimate_gradient_gsfa

numberOfRuns = 20000
def RunExperiment(p, numRuns, STOCHASTIC, MU, SIGMA, i):
    np.random.seed(i)
    n = 3
    thresholds = np.array([2, 3, 1])
    rho = 0.6

    V = np.random.standard_normal(numRuns)

    W = np.random.exponential(1 / 0.3, numRuns) # Should we choose 0.3 or 1/0.3?

    eta = np.zeros((n, numRuns))
    X = np.zeros((n, numRuns))
    Y = np.zeros((n, numRuns))
    returns = np.zeros((n, numRuns))
    for i in range(n):
        eta[i, :] = np.random.normal(0, scale=np.sqrt(i + 1), size=numRuns)
        X[i, :] = (rho * V + np.sqrt(1 - rho ** 2) * eta[i,:]) / np.maximum(W, 1)
        Y[i, :] = np.random.uniform(low=0, high=X[i, :], size=(1, numRuns))

        indicator = (X[i, :] >= thresholds[i]).astype(int)
        returns[i, :] = p[i] *  Y[i, :] * indicator

    sum = np.sum(returns, axis=0)
    std = np.std(sum)
    return sum, std

### For projected gradient descent, we need to evaluate points that lie outside of the constraint set
def Objective1(p, STOCHASTIC, MU, SIGMA, i):
    sum, std = RunExperiment(p, numberOfRuns, STOCHASTIC, MU, SIGMA, i)

    if np.mean(sum) == 0:
        return 0

    return np.mean(sum) / std


def project_onto_simplex(x):
    # Check if x is already in the simplex
    if np.all(x >= 0) and np.sum(x) == 1:
        return x

    # Find the projection of x onto the simplex
    n = len(x)
    u = np.sort(x)[::-1]
    cumsum = np.cumsum(u)

    y = np.zeros(len(x))

    for j in range(len(x)):
        y[j] = u[j] + 1 / (j + 1) * (1 - cumsum[j])

    rho = np.nonzero(y > 0)[0][-1]

    l = 1 / (rho + 1) * (1 - np.sum(u[:rho + 1]))
    z = np.maximum(x + l, 0)

    if abs((np.sum(z) - 1)) > 1e-9:
        print("x = ", x)
        print("z = ", z)
        print("Sum z = ", np.sum(z))
    return z




p_0 = np.array([1,0,0])
EPSILON_TYPE = 'decreasing'  # Use 'fixed' or 'decreasing' for ε
EPSILON_VALUE = 0.001  # Initial value of ε if EPSILON_TYPE is 'fixed'
NR_ITERATIONS = 20  # Number of iterations
STOCHASTIC = False  # Set to True if you want to add stochasticity
MU = 0.0  # Mean of the stochastic noise
SIGMA = 0.1  # Standard deviation of the stochastic noise
BATCH = False  # Set to True for batch estimation of the gradient
NR_ESTIMATES = 3  # Number of estimates when BATCH is True
OPTIMIZATION_TYPE = 'maximization'  # 'minimization' or 'maximization'


thetas, gradients, objective_values = GradientDescent(Objective1, estimate_gradient_gsfa, p_0, EPSILON_TYPE, EPSILON_VALUE, NR_ITERATIONS, STOCHASTIC, MU, SIGMA, BATCH, NR_ESTIMATES, OPTIMIZATION_TYPE, project_onto_simplex)

print(thetas[-1, :])

plt.figure()
plt.plot(range(NR_ITERATIONS), objective_values)

plt.show()
