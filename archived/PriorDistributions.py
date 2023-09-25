import numpy as np
import pandas as pd

from scipy.stats import binom, poisson

class PriorDistributionArrays:

    """
    Creates arrays of the prior distributions, for given observed variables, in the format required for the BaumWelch class.

    Attributes
    ----------
    df : pd.DataFrame
        Data frame of time series data.

    Methods
    -------
    binomial_distribution
        Creates a binomial distribution, with parameters n and p, for a given observed discrete variable.
    poisson_distribution
        Creates a Poisson distribution, with a given mean, for an observed discrete variable.
    concatenate_B_array
        Given a list of prior probability distribution arrays, they are concatenated into one array, where the zeroth dimension corresponds to a hidden state.
    """

    def __init__(self, df: pd.DataFrame):

        self.df = df

    def O_min_max_vals(self, observable : str):

        """
        Find the min, max, and dimensions of the range for a given observed discrete variable.

        Parameters
        ----------
        observable : str
            Name of feature wish to make the Poisson prior from.

        Return
        ------
        K : int
            Dimensions of range of observed variable.
        K_min : int
            Minimum value of variable.
        K_max : int
            Maximal value of variable.
        """

        # find min and max vals and dims of range
        K_max = self.df[observable].max() 
        K_min = self.df[observable].min()
        K = (K_max - K_min) + 1

        return K, K_min, K_max
    
    def B_instantiate(self, observable : str):

        """
        Given an observed variable, find the range of values and instantiate the B prior probability array for 1 hidden state.

        Parameters
        ----------
        observable : str
            Name of feature wish to make the Poisson prior from.

        Return
        ------
        B : np.array
            Instantiated prior probability array of dims (1,K,2). B[0,:,0] is populated with the relevant range of observed variable values.
        """

        # find range of vals
        K, K_min, K_max = self.O_min_max_vals(observable)

        # probability array: instantiate and populate with range of observable vals
        B = np.zeros((1,K,2))
        B[0,:,0] = np.arange(start=K_min, stop=K_max+1, dtype=np.int32)

        return B

    def binomial_distribution(self, observable : str, n : int, p : float):

        """
        Creates a Binomial probability distribution for an observed variable, given n and p. The returned array is of dims (1,K,2), where K is the number of possible observed values.
        The row in the first leaf contains the range of possible observed values, and the second leaf the concomitant emission probabilities.

        Parameters
        ----------
        observable : str
            Name of feature wish to make the Poisson prior from.
        n : int
            Size of sample.
        p : float [0.0, 1.0]
            Probability of success.

        Return
        ------
        B : np.array
            Binomial prior probabilites returned in second leaf. Array of dims (1,K,2).
        """

        # instantiate the B prior probability array
        B = self.B_instantiate(observable)

        # now find probability emissions from the Poisson pmf
        for i, oi in enumerate(B[0,:,0]):
            B[0,i,1] = binom.pmf(oi, n, p)

        return B

    def poisson_distribution(self, observable : str, mu : float):

        """
        Creates a Poisson probability distribution from the mean of an observed variable. The returned array is of dims (1,K,2), where K is the number of possible observed values.
        The row in the first leaf contains the range of possible observed values, and the second leaf the concomitant emission probabilities.

        Parameters
        ----------
        observable : str
            Name of feature wish to make the Poisson prior from.
        mu : float
            Mean for the Poisson distribution.

        Return
        ------
        B : np.array
            Poisson prior probabilites returned in second leaf. Array of dims (1,K,2).
        """

        # instantiate the B prior probability array
        B = self.B_instantiate(observable)

        # now find probability emissions from the Poisson pmf
        for i, oi in enumerate(B[0,:,0]):
            B[0,i,1] = poisson.pmf(oi, mu)

        return B
    
    @ staticmethod
    def concatenate_B_array(B_arrays_list : list):

        """
        Given a list of prior probability arrays of equal dimensions, each corresponding to a hidden state, a single, concatenated array, in which each row is a hidden state, is returned.

        Parameters
        ----------
        B_arrays_list : list
            The prior probability distribution for each hidden state, ordered. Each array is of dims (1,K,2). List is of length N, corresponding to the number of hidden states.

        Return
        ------
        B : np.array
            Poisson prior probabilites returned in second leaf. Array of dims (N,K,2), where N is the number of hidden states.
        """

        # check all arrays of same dims
        for i, B in enumerate(B_arrays_list):

            if i == 0:
                N, K, L = B.shape
                continue
            else:
                assert N == B.shape[0], "Dimension 0 of B array does not match."
                assert K == B.shape[1], "Dimension 1 of B array does not match."
                assert L == B.shape[2], "Dimension 2 of B array does not match."
                N, K, L = B.shape

        # now re-assign N to length of list and instantiate array
        N = len(B_arrays_list)
        B = np.zeros((N,K,2))

        # fill in the array
        for Ni, Bi in enumerate(B_arrays_list):
            B[Ni, :, :] = Bi

        return B
 
