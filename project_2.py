import numpy as np
from SGDFOaL.functions import GradientDescent, estimate_gradient_spsa, estimate_gradient_gsfa

numberOfRuns = 100000

def RunExperiment(theta, numRuns, STOCHASTIC, MU, SIGMA):
    ###
    if theta < 0:
        return "Invalid theta"

    N = 10

    interArrivalTimes = np.random.exponential(5, size=(N, numRuns))
    serviceTimes = np.random.exponential(theta, size=(N, numRuns))

    # Calculate the waiting times:
    waitingTimes = np.zeros((N, numRuns))
    for i in range(N - 1):
        waitingTimes[i + 1, :] = np.maximum(0, waitingTimes[i, :] + serviceTimes[i, :] - interArrivalTimes[i, :])

    # Caclulate the average waiting time for each customer
    averageWaitingTimes = np.mean(waitingTimes, axis=0)

    # The 90% quantile of the average waiting time:
    quantile = np.quantile(averageWaitingTimes, 0.9)

    return quantile

def Objective(theta, STOCHASTIC, MU, SIGMA):
    z = 8

    quantile = RunExperiment(theta, numberOfRuns, STOCHASTIC, MU, SIGMA)

    # maybe we could also first take the mean and then compute the squared difference?
    # this is what I would find logical
    return np.mean((quantile - z) ** 2)



theta_0 = np.array([10])
EPSILON_TYPE = 'fixed'  # Use 'fixed' or 'decreasing' for ε
EPSILON_VALUE = 0.01  # Initial value of ε if EPSILON_TYPE is 'fixed'
NR_ITERATIONS = 500  # Number of iterations
STOCHASTIC = False  # Set to True if you want to add stochasticity
MU = 0.0  # Mean of the stochastic noise
SIGMA = 0.1  # Standard deviation of the stochastic noise
BATCH = True  # Set to True for batch estimation of the gradient
NR_ESTIMATES = 3  # Number of estimates when BATCH is True
OPTIMIZATION_TYPE = 'minimization'  # 'minimization' or 'maximization'

thetas, gradients, objective_values = GradientDescent(Objective, estimate_gradient_gsfa, theta_0, EPSILON_TYPE, EPSILON_VALUE, NR_ITERATIONS, STOCHASTIC, MU, SIGMA, BATCH, NR_ESTIMATES, OPTIMIZATION_TYPE)
print(thetas[-1])