import numpy as np
from SGDFOaL.functions import GradientDescent, estimate_gradient_spsa, estimate_gradient_gsfa, \
    estimate_gradient_finite_differences

numberOfRuns = 50

def RunExperiment(theta, numRuns, STOCHASTIC, MU, SIGMA, i):
    np.random.seed(i)
    if theta[0] < 0:
        #raise Exception("theta should be positive")
        theta[0] = 0.00000001

    N = 10

    interArrivalTimes = np.random.exponential(5, size=(N, numRuns))
    serviceTimes = np.random.exponential(theta[0], size=(N, numRuns))

    # Calculate the waiting times:
    waitingTimes = np.zeros((N, numRuns))
    for j in range(N - 1):
        waitingTimes[j + 1, :] = np.maximum(0, waitingTimes[j, :] + serviceTimes[j, :] - interArrivalTimes[j + 1, :])

    # Caclulate the average waiting time for each customer
    averageWaitingTimes = np.mean(waitingTimes, axis=0)

    return averageWaitingTimes

def Objective(theta, STOCHASTIC, MU, SIGMA, i):
    z = 8

    quantile = np.quantile(RunExperiment(theta, numberOfRuns, STOCHASTIC, MU, SIGMA, i), 0.9)

    # maybe we could also first take the mean and then compute the squared difference?
    # this is what I would find logical
    return np.mean((quantile - z) ** 2)

def SimpleGradient(objective_f, theta, i, STOCHASTIC, MU, SIGMA, BATCH, NR_ESTIMATES, initial_seed = 0):
    np.random.seed(i + initial_seed)
    experiment = RunExperiment(theta, numberOfRuns, STOCHASTIC, MU, SIGMA, i)

    # amount of times waiting time is above 8
    indicator = (experiment <= theta[1]).astype(int)

    q_update = 10 * (0.9 - np.mean(indicator))

    theta_update = -2 * (theta[1] - 8)

    return -1 * np.array([theta_update, q_update])





theta_0 = np.array([3, 7])
EPSILON_TYPE = 'decreasing'  # Use 'fixed' or 'decreasing' for ε
EPSILON_VALUE = 0.01  # Initial value of ε if EPSILON_TYPE is 'fixed'
NR_ITERATIONS = 5000  # Number of iterations
STOCHASTIC = False  # Set to True if you want to add stochasticity
MU = 0.0  # Mean of the stochastic noise
SIGMA = 0.1  # Standard deviation of the stochastic noise
BATCH = False  # Set to True for batch estimation of the gradient
NR_ESTIMATES = 3  # Number of estimates when BATCH is True
OPTIMIZATION_TYPE = 'minimization'  # 'minimization' or 'maximization'

thetas, gradients, objective_values = GradientDescent(Objective, SimpleGradient, theta_0, EPSILON_TYPE, EPSILON_VALUE, NR_ITERATIONS, STOCHASTIC, MU, SIGMA, BATCH, NR_ESTIMATES, OPTIMIZATION_TYPE)
print(thetas[-1])

print("Initial starting point: ", theta_0)
print("End point: ", thetas[-1])