import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3010)

THETA = 15
MIN_C = 1
MAX_C = 15
COMMON_RN = False
NR_SAMPLES = [10,100,1000,10000]

num_of_values = 101
c = np.linspace(MIN_C,MAX_C,num_of_values)
Y_expectation = np.zeros(num_of_values)
Y_variance = np.zeros(num_of_values)
Y_expectation_common = np.zeros(num_of_values)
Y_variance_common = np.zeros(num_of_values)

    
fig,ex = plt.subplots(1,2, figsize=(16, 5))
fig.patch.set_facecolor("gainsboro")

for samples in NR_SAMPLES:
    for i in range(0,num_of_values):
        U = np.random.uniform(0, 1, samples)
        X_plus = -50 * (THETA + c[i]) * np.log(1-U)
        X_substract = -50 * (THETA - c[i]) * np.log(1-U)
    
        Y = 1 - 1 / (2*c[i]) * (X_plus / np.sqrt(X_plus+1)
                                - X_substract / np.sqrt(X_substract+1))
    
        Y_expectation_common[i] = np.mean(Y)
        Y_variance_common[i] = np.var(Y)
    
        U = np.random.uniform(0, 1, samples)
        X_plus = -50 * (THETA + c[i]) * np.log(1-U)
    
        U = np.random.uniform(0, 1, samples)
        X_substract = -50 * (THETA - c[i]) * np.log(1-U)
    
        Y = 1 - 1 / (2*c[i]) * (X_plus / np.sqrt(X_plus+1)
                                - X_substract / np.sqrt(X_substract+1))
    
        Y_expectation[i] = np.mean(Y)
        Y_variance[i] = np.var(Y)
    
    ex[0].set_title("Expectation of $Y(theta,c)$")
    ex[0].set_xlabel("$c$")
    ex[0].grid(True)
    ex[0].plot(c, Y_variance,label = "a, |u|= "+str(samples))
    ex[0].plot(c, Y_variance_common,label = "b, |u|= "+str(samples),linestyle ='--')
    ex[0].legend()
    ex[0].set_ylim([0,10])
    
    #Plot variance figure
    ex[1].set_title("Variance of $Y(theta,c)$")
    ex[1].set_xlabel("$c$")
    ex[1].grid(True)
    ex[1].plot(c, Y_variance,label = "a, |u|= "+str(samples))
    ex[1].plot(c, Y_variance_common,label = "b, |u|= "+str(samples),linestyle ='--')
    ex[1].legend()
    ex[1].set_ylim([0,20])
    
    fig.savefig("Ex_mining.png")