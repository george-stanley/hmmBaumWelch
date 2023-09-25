import pytest
import numpy as np
from hmmBaumWelch import BaumWelch

# define class attributes as the data to be used throughout the testing class
N=2 # number of hidden states
T=3 # number of time points in data
K=3 # number of different values of a variable - deliberately set to equal T for both variables
R=2 # number of observed variables

# observed variables
O0 = [0, 4, 1]
O1 = [-0.1, 0.9, 0.1]

# the list of observed variables
O_R = [O0, O1]

# priors
B0 = np.zeros((N,K,2), dtype=np.float64)
B1 = np.zeros((N,K,2), dtype=np.float64)

# for Z_{i}=0
B0[0,:,0] = [0, 4, 1] # observed vals in first leaf
B0[0,:,1] = [0.4, 0.025, 0.3] # concomitant probs in second

# repeat for Z_{i}=1
B0[1,:,0] = [0, 4, 1] # observed vals in first leaf
B0[1,:,1] = [0.05, 0.4, 0.05] # concomitant probs in second

# repeat for prior of second observable
# for Z_{i}=0
B1[0,:,0] = [-0.1, 0.9, 0.1] # observed vals in first leaf
B1[0,:,1] = [0.25, 0.075, 0.15] # concomitant probs in second

# repeat for Z_{i}=1
B1[1,:,0] = [-0.1, 0.9, 0.1] # observed vals in first leaf
B1[1,:,1] = [0.10, 0.25, 0.10] # concomitant probs in second

# add together in list
B_R = [B0, B1]

A = np.array([[0.80, 0.20],
            [0.20, 0.80]], dtype=np.float64)
pi = [0.50, 0.50]
observables_weights = [1,1] # treat each variable equally
Z = (0,1) # set of hidden states

@pytest.fixture
def run_BaumWelch():
    """
    Runs the initiates the BaumWelch object and runs 1 iteration of expectation maximisation.
    The results are used in testing.
    """
    # create the HMM object
    HMM = BaumWelch(
        Z,
        O_R,
        pi,
        A,
        B_R,
        observables_weights,  # Use the default observables_weights
    )

    # perform expectation maximisation
    HMM.baumwelch_expectationMaximisation(iter=1, update_pi=False, update_A=False, update_B=False, early_stopping=False)

    return HMM

def test_forwards_compute(run_BaumWelch):

    """
    Test the forwards compute function, producing alpha_log, without using the "log-sum-exp" trick.

    - alpha_{i}(t) = P(o_{1:t}, Z_{t} | theta), and where theta = (A,B,pi).
    """

    # get the HMM object from the fixture
    HMM = run_BaumWelch

    # instantiate the alpha array
    alpha = np.zeros((N,T,R), dtype=np.float64)

    # manually do the calculations without using the "log-sum-exp" trick
    for r in range(R):
        for t in range(T):
            for i in range(N):

                if t==0: # initiation step
                    alpha[i,t,r] = pi[i]*B_R[r][i,t,1]
                else:
                    alpha[i,t,r] =  B_R[r][i,t,1]*(alpha[0,t-1,r]*A[0,i] + alpha[1,t-1,r]*A[1,i])

    # take natural logarithm of the alpha array
    alpha_log = np.log(alpha)

    # compare with results from BaumWelch package
    assert np.allclose(HMM.alpha_log, alpha_log, rtol=1e-06, atol=1e-08, equal_nan=False), "`alpha_log` array from `forwards_compute` is not matching."

def test_backwards_compute(run_BaumWelch):

    """
    Test the backwards compute function, producing beta_log, without using the "log-sum-exp" trick.

    - beta_{i}(t) = P(o_{t+1:T} | Z_{t}=i, theta), and where theta = (A,B,pi).
    """

    # get the HMM object from the fixture
    HMM = run_BaumWelch

    # instantiate
    beta = np.zeros((N,T,R), dtype=np.float64)

    # manually do the calculations without using the "log-sum-exp" trick
    for r in range(R):
        for t in range(T-1,-1,-1): # iterating backwards
            for i in range(N):

                if t==T-1: # initiation
                    beta[i,t,r] = 1.0
                else:
                    beta[i,t,r] = (beta[0,t+1,r]*A[i,0]*B_R[r][0,t+1,1] + beta[1,t+1,r]*A[i,1]*B_R[r][1,t+1,1])

    # take natural logarithm
    beta_log = np.log(beta)

    assert np.allclose(HMM.beta_log, beta_log, rtol=1e-06, atol=1e-08, equal_nan=False), "`beta_log` array from `backwards_compute` is not matching."

def test_gamma_compute(run_BaumWelch):

    """
    Test the gamma compute function, without using the "log-sum-exp" trick.

    - gamma_{i}(t) = P(Z_{t}=i | O, theta) = P(Z_{t}=i, O | theta) / P(O | theta)
    """

    # get the HMM object from the fixture
    HMM = run_BaumWelch

    # extract and eponentiate the log(alpha) and log(beta) arrays 
    alpha = np.exp(HMM.alpha_log)
    beta  = np.exp(HMM.beta_log)

    # instantiate
    gamma = np.zeros((N,T,R), dtype=np.float64)

    # manually do the calculations with exponentiated arrays, therefore not using the "log-sum-exp" trick
    for r in range(R):
        for t in range(T):
            for i in range(N):
                gamma[i,t,r] = (alpha[i,t,r]*beta[i,t,r])/(alpha[0,t,r]*beta[0,t,r] + alpha[1,t,r]*beta[1,t,r])

    assert np.allclose(HMM.gamma, gamma, rtol=1e-06, atol=1e-08, equal_nan=False), "`gamma` array from `gamma_compute` is not matching."

def test_xi_compute(run_BaumWelch):

    """
    Test the xi compute function.

    - xi_{ij}(t) = P(Z_{t}=i, Z_{t+1}=j| O, theta) = P(Z_{t}=i, Z_{t+1}=j, O | theta) / P(O | theta)
    """

    # get the HMM object from the fixture
    HMM = run_BaumWelch

    # extract and eponentiate the log(alpha) and log(beta) arrays 
    alpha = np.exp(HMM.alpha_log)
    beta  = np.exp(HMM.beta_log)

    # instantiate: expected dims=(N, N, T-1, R)
    xi = np.zeros((N,N,T-1,R), dtype=np.float64)

    # manually do the calculations for O_{r=0} with out using logs or the "log-sum-exp" trick
    for r in range(R):
        for t in range(T-1):
            for i in range(N):
                for j in range(N):

                    # compute numerator
                    numerator = alpha[i,t,r]*A[i,j]*beta[j,t+1,r]*B_R[r][j,t+1,1]

                    # now denominator
                    denominator=0
                    for k in range(N):
                        for w in range(N):
                            denominator += alpha[k,t,r]*A[k,w]*beta[w,t+1,r]*B_R[r][w,t+1,1]

                    xi[i,j,t,r] = numerator/denominator

    assert np.allclose(HMM.xi, xi, rtol=1e-06, atol=1e-08, equal_nan=False), "`gamma` array from `gamma_compute` is not matching."

@pytest.mark.parametrize("observables_weights", [[100, 0], [75, 25], [50, 50], [25, 75], [0, 100]])
def test_expectationMaximisation_inference(observables_weights):

    """
    Test the weighted averaging of gamma for performing inference.
    """

    # create HMM object with different observables_weights
    HMM = BaumWelch(
        Z,
        O_R,
        pi,
        A,
        B_R,
        observables_weights=observables_weights,
    )

    # check weights have been automatically normalised
    assert sum(HMM.observables_weights)==1, "BaumWelch class should normalise all weights values such that their sum equals 1.0."

    # perform expectation maximisation
    HMM.baumwelch_expectationMaximisation(iter=1, update_pi=False, update_A=False, update_B=False, early_stopping=False)

    # perform inference - the class should automatically use weights specify at instantiation
    _, _, _, gamma_meanR = HMM.Z_state_probs_inference()

    # extract the gamma array
    gamma = HMM.gamma

    # instantiate gamma arrays for manually computing weighted gamma arrays
    gamma_meanR_test_weighted     = np.zeros((N,T), dtype=np.float64)
    gamma_meanR_test_weighted_ave = np.zeros((N,T), dtype=np.float64)

    # manually normalise the weighting arrays
    weights_min      = min(observables_weights)
    weights_max      = max(observables_weights)

    # if-else statement to avoid division by 0
    if weights_min == 0:
        weights_norm = [weight/weights_max for weight in observables_weights]
    else:
        weights_norm = [weight/weights_min for weight in observables_weights]

    # normalise
    weights_norm_sum = sum(weights_norm)
    weights_norm     = [weight/weights_norm_sum for weight in weights_norm]

    # compute weighted gamma quanity for all Zi and t
    for i in range(N):
        for t in range(T):
            gamma_meanR_test_weighted[i,t] = (gamma[i,t,0]*weights_norm[0] + gamma[i,t,1]*weights_norm[1])

    # compute using Numpy's average func (should be the same)
    gamma_meanR_test_weighted_ave = np.average(gamma, axis=-1, weights=observables_weights)

    # check all these quantities against those calculated within the BaumWelch package
    assert np.allclose(gamma_meanR, gamma_meanR_test_weighted, rtol=1e-06, atol=1e-08, equal_nan=False)
    assert np.allclose(gamma_meanR, gamma_meanR_test_weighted_ave, rtol=1e-06, atol=1e-08, equal_nan=False)
