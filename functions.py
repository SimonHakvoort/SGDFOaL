# import libraries
import numpy as np

# Adds normally distributed noise to the objective function.
def add_noise(MU, SIGMA):
    return np.random.normal(MU, SIGMA, size=1)

def no_constraint(theta):
    return theta

def estimate_gradient_spsa(objective_f, theta, i, STOCHASTIC, MU, SIGMA, BATCH, NR_ESTIMATES):
    delta_i = np.random.choice((-1, 1), size=theta.shape)
    eta_i = 1 / (i + 1)

    if BATCH:
        gradient_estimates = np.zeros((NR_ESTIMATES, len(theta)))
        for n in range(NR_ESTIMATES):
            perturbation_high = objective_f(theta + eta_i * delta_i, STOCHASTIC, MU, SIGMA)
            perturbation_low = objective_f(theta - eta_i * delta_i, STOCHASTIC, MU, SIGMA)
            numerator = perturbation_high - perturbation_low
            denominator = 2 * eta_i * delta_i
            gradient_estimates[n,:] = numerator / denominator
            delta_i = np.random.choice((-1, 1), size=theta.shape)
        gradient_estimate = np.mean(gradient_estimates, axis=0)
    else:
        perturbation_high = objective_f(theta + eta_i * delta_i, STOCHASTIC, MU, SIGMA)
        perturbation_low = objective_f(theta - eta_i * delta_i, STOCHASTIC, MU, SIGMA)
        numerator = perturbation_high - perturbation_low
        denominator = 2 * eta_i * delta_i
        gradient_estimate = numerator / denominator

    return gradient_estimate

def estimate_gradient_gsfa(objective_f, theta, i, STOCHASTIC, MU, SIGMA, BATCH, NR_ESTIMATES):
    delta_i = np.random.randn(len(theta))
    eta_i = 1 / (i + 1)

    if BATCH:
        gradient_estimates = np.zeros((NR_ESTIMATES, len(theta)))
        for n in range(NR_ESTIMATES):
            gradient_estimates[n,:] = delta_i / eta_i * (objective_f(theta + delta_i * eta_i, STOCHASTIC, MU, SIGMA) - objective_f(theta, STOCHASTIC, MU, SIGMA))
            delta_i = np.random.randn(len(theta))
        gradient_estimate = np.mean(gradient_estimates, axis=0)
    else:
        gradient_estimate = delta_i / eta_i * (objective_f(theta + delta_i * eta_i, STOCHASTIC, MU, SIGMA) - objective_f(theta, STOCHASTIC, MU, SIGMA))
    return gradient_estimate

# The SPSA algorithm
def GradientDescent(objective_f, gradient_estimator, THETA_0, EPSILON_TYPE, EPSILON_VALUE, NR_ITERATIONS, STOCHASTIC, MU, SIGMA, BATCH, NR_ESTIMATES, OPTIMIZATION_TYPE, projection = no_constraint):
    thetas = np.zeros((NR_ITERATIONS + 1, len(THETA_0)))
    gradients = np.zeros((NR_ITERATIONS, len(THETA_0)))
    objective_values = np.zeros(NR_ITERATIONS)
    thetas[0, :] = THETA_0

    for i in range(NR_ITERATIONS):
        g = gradient_estimator(objective_f, thetas[i, :], i, STOCHASTIC, MU, SIGMA, BATCH, NR_ESTIMATES)
        gradients[i] = g
        if EPSILON_TYPE == 'fixed':
            if OPTIMIZATION_TYPE == 'minimization':
                thetas[i + 1,:] = thetas[i,:] - EPSILON_VALUE * g
            if OPTIMIZATION_TYPE == 'maximization':
                thetas[i + 1,:] = thetas[i,:] + EPSILON_VALUE * g
        if EPSILON_TYPE == 'decreasing':
            if OPTIMIZATION_TYPE == 'minimization':
                thetas[i + 1,:] = thetas[i,:] - 1 / (i + 100) * g
            if OPTIMIZATION_TYPE == 'maximization':
                thetas[i + 1,:] = thetas[i,:] + 1 / (i + 100) * g
        thetas[i + 1, :] = projection(thetas[i + 1, :])
        objective_values[i] = objective_f(thetas[i + 1], STOCHASTIC, MU, SIGMA)
        print(thetas[i + 1])
        print(objective_values[i])

    return thetas, gradients, objective_values

def Objective_J(theta, STOCHASTIC, MU, SIGMA):
    first_part = theta[0] ** 4 - 16 * theta[0] ** 2 + 5 * theta[0]
    second_part = theta[1] ** 4 - 16 * theta[1] ** 2 + 5 * theta[1]

    if STOCHASTIC:
        return 0.5 * (first_part + second_part) + add_noise(MU, SIGMA)

    return 0.5 * (first_part + second_part)
