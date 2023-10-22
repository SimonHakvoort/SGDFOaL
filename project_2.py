import numpy as np
from SGDFOaL.functions import GradientDescent, estimate_gradient_spsa, estimate_gradient_gsfa

numberOfRuns = 100

def RunExperiment(p, numRuns, STOCHASTIC, MU, SIGMA):
    N = 10
    z = 8

    interArrivalTimes = np.random.exponential(5, size=(N, numRuns))
