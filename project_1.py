import numpy as np

STOCHASTIC = False
MU = 0
SIGMA = 1
numberOfRuns = 100
def RunExperiment(p, numRuns, STOCHASTIC, MU, SIGMA):
    n = 3
    thresholds = np.array([2, 3, 1])
    rho = 0.6

    V = np.random.standard_normal(numRuns)

    W = np.random.exponential(1 / 0.3, numRuns)

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

    sum = np.mean(returns, axis=0)
    std = np.std(sum)
    return sum, std


def Objective(p, STOCHASTIC, MU, SIGMA):
    if np.abs(np.sum(p) - 1) >= 1e-9:
        print(np.sum(p))
        return "Hard constraint not satisfied"

    if np.any(p < 0):
        print(p)
        return "Hard constraint not satisfied"

    sum, std = RunExperiment(p, numberOfRuns, STOCHASTIC, MU, SIGMA)

    return np.mean(sum) / std


def project_onto_simplex(x):
    # Check if x is already in the simplex
    if np.all(x >= 0) and np.sum(x) == 1:
        return x

    # Find the projection of x onto the simplex
    n = len(x)
    u = np.sort(x)[::-1]
    cumsum = np.cumsum(u)
    rho = np.where(u > (cumsum - 1) / np.arange(1, n + 1))[0][-1]
    theta = np.max([0, (cumsum[rho] - 1) / (rho + 1)])
    projection = np.maximum(x - theta, 0)
    return projection

x = np.array([1.1, 3, 0])

print(project_onto_simplex(x))