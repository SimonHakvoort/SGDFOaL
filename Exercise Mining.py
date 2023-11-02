import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3010)

THETA = 15
MIN_C = 1
MAX_C = 15
NR_SAMPLES = np.linspace(1, 6, 6)

num_of_values = 101
c = np.linspace(MIN_C, MAX_C, num_of_values)
Y_expectation = np.zeros(num_of_values)
Y_variance = np.zeros(num_of_values)
Y_expectation_common = np.zeros(num_of_values)
Y_variance_common = np.zeros(num_of_values)

fig, ex = plt.subplots(2, 3, figsize=(16, 8))
fig.patch.set_facecolor("gainsboro")

for samples in NR_SAMPLES:
    for i in range(0, num_of_values):
        U = np.random.uniform(0, 1, int(10 ** samples))
        X_plus = -50 * (THETA + c[i]) * np.log(1 - U)
        X_substract = -50 * (THETA - c[i]) * np.log(1 - U)

        Y = 1 - 1 / (2 * c[i]) * (X_plus / np.sqrt(X_plus + 1)
                                  - X_substract / np.sqrt(X_substract + 1))

        Y_expectation_common[i] = np.mean(Y)
        Y_variance_common[i] = np.var(Y)

        U = np.random.uniform(0, 1, int(10 ** samples))
        X_plus = -50 * (THETA + c[i]) * np.log(1 - U)

        U = np.random.uniform(0, 1, int(10 ** samples))
        X_substract = -50 * (THETA - c[i]) * np.log(1 - U)

        Y = 1 - 1 / (2 * c[i]) * (X_plus / np.sqrt(X_plus + 1)
                                  - X_substract / np.sqrt(X_substract + 1))

        Y_expectation[i] = np.mean(Y)
        Y_variance[i] = np.var(Y)

    ex[0][0].set_title("(a.) Expectation Independent Samples of Y(θ,c)")
    ex[0][0].grid(True)
    ex[0][0].plot(c, Y_expectation, label="$n= 10^{" + str(int(samples)) + "}$")
    ex[0][0].legend(loc='upper right')
    ex[0][0].set_xlim([0, 22])

    # Plot variance figure
    ex[1][0].set_title("(a.) Variance Independent Samples of Y(θ,c)")
    ex[1][0].set_xlabel("$c$")
    ex[1][0].grid(True)
    ex[1][0].plot(c, Y_variance, label="$n= 10^{" + str(int(samples)) + "}$")
    ex[1][0].legend(loc='upper right')
    ex[1][0].set_xlim([0, 22])

    ex[0][1].set_title("(b.) Expectation Dependent Samples of Y(θ,c)")
    ex[0][1].grid(True)
    ex[0][1].plot(c, Y_expectation_common, label="$n= 10^{" + str(int(samples)) + "}$")
    ex[0][1].legend(loc='upper right')
    ex[0][1].set_xlim([0, 22])

    # Plot variance figure
    ex[1][1].set_title("(b.) Variance Dependent Samples of Y(θ,c)")
    ex[1][1].set_xlabel("$c$")
    ex[1][1].grid(True)
    ex[1][1].plot(c, Y_variance_common, label="$n= 10^{" + str(int(samples)) + "}$")
    ex[1][1].legend(loc='upper right')
    ex[1][1].set_xlim([0, 22])

    ex[0][2].set_title("Difference Expectation (a.),(b.) Y(θ,c)")
    ex[0][2].grid(True)
    ex[0][2].plot(c, np.abs(Y_expectation - Y_expectation_common), label="$n= 10^{" + str(int(samples)) + "}$")
    ex[0][2].legend(loc='upper right')
    # ex[0][2].set_xlim([0,22])

    ex[1][2].set_title("Difference variance (a.),(b.) Y(θ,c)")
    ex[1][2].grid(True)
    ex[1][2].plot(c, np.abs(Y_variance - Y_variance_common), label="$n= 10^{" + str(int(samples)) + "}$")
    ex[1][2].legend(loc='upper right')
    # ex[1][2].set_xlim([0,22])
    ex[1][2].set_ylim([0, 5])
    ex[1][2].set_xlabel("$c$")

    fig.savefig("Ex_mining.png")