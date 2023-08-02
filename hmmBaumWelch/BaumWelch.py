import numpy as np

class BaumWelch:

    """
    Builds a model object for implementing the Baum-Welch expectation maximisation algorithm to maximise:

        - theta^{*} = argmax_{theta} P(O_{1:R} | theta),

    where theta represents the hidden Markov model (HMM) parameters: the hidden state transition probabilities, A; the obervable variables emission probabilities - or priors -
    B; and the initial state probabilities, pi. Therefore:

        - theta = (A, B, pi).

    The Baum-Welch algorithm also performs hidden state inference.

    Attributes
    ----------
    Z : set[int]
        The set of possible hidden Markov states (i.e. Z={0,1})
    O_list : list[list[int]]
        List of R observed variables, i.e. R=number of observed variables.
    pi : list[float]
        Probabilities of being in each hidden Markov state at t=1.
    A : np.ndarray[float]
        Transmission probabilities matrix, of dims (N,N), where N=number of hidden states.
    B_list : list[np.ndarray[float]]
        List of emission probability matrices, each of dims (N,K_{r}), where K_{r}=number of different possible observations for observed variable O_{r}.
    observables_weights : np.ndarray, default=None
        Weighted values for the observable variables when performing inference (note, these values do not affect the transition probability matrix, A).
    N : int
        Number of hidden states.
    T : int
        Number of observed time points in sequence.
    R : int
        Number of different observed variables.
    gamma : np.ndarray[float]
        P(Z_{t}=i | O, theta); dims=(N,T).
    xi : np.ndarray[float]
        P(Z_{t}=i, Z_{t+1}=j| O, theta); dims=(N, N, T-1).
    Z0_probs : np.ndarray[float]
            P(Z_{i} when i=0 | O_{1}, ..., O_{R}, theta); expected dims: (iter,T), where iter is number of iterations used in expectation maximisation.
    Z1_probs : np.ndarray[float]
            P(Z_{i} when i=1 | O_{1}, ..., O_{R}, theta); expected dims: (iter,T).
    Z_inferences : np.ndarray[int]
            Inferred hidden state sequence; expected dims: (iter,T).
    gamma_meanR : np.ndarray[float]
            P(Z_{i} where i=row | O_{1}, ..., O_{R}, theta); expected dims: (N,T).

    Methods
    -------
    forwards_backwards_algorithm : tuple[np.ndarray[float]]
        Implements the forward-backwards algorithm to return gamma and xi.
    Z_state_probs_inference : tuple[np.ndarray]
        Given gamma array of dims (N, T, R), averages the results along the R axis (i.e., across all observables), and return the Z inference probabilities and states.
    baumwelch_expectationMaximisation : tuple[np.ndarray[float]]
        Implements the Baum-Welch algorithm, i.e., the forwards-backwards followed by HMM parameter updates (theta), a given number of times (defined by `iter`).
        Also returns the hidden state inference.
    """

    def __init__(
        self,
        Z: set[int],
        O_list: list[list[int]],
        pi: list[float],
        A: np.ndarray,
        B_list: list[np.ndarray],
        observables_weights: list[float] = None,
    ):

        """
        Creates the hidden Markov model object.

        Parameters
        ----------
        Z : set[int]
            The set of possible hidden Markov states (i.e. Z={0,1})
        O_list : list[list[int]]
            List of R observed variables, i.e. R=number of observed variables.
        pi : list[float]
            Probabilities of being in each hidden Markov state at t=1.
        A : np.ndarray[float]
            Transmission probabilities matrix, of dims (N,N), where N=number of hidden states.
        B_list : list[np.ndarray[float]]
            List of emission probability matrices, each of dims (N,K_{r}), where K_{r}=number of different possible observations for observed variable O_{r}.
        observables_weights : list[float], default=None
            Weighted values for the observable variables when performing inference.
        """

        # store HMM parameters as instance attributes
        self.Z = Z
        self.O_list = O_list
        self.pi = pi
        self.A = A
        self.B_list = B_list

        # other useful constants to store as attributes
        self.N = len(Z)
        self.T = len(O_list[0])
        self.R = len(O_list)

        # if observables weights is not none, scale and turn into array
        if observables_weights is not None:

            # first, assert of correct dimensions
            try:
                assert self.R == len(
                    observables_weights
                ), "Length(observables_weights) does not match number of observed variables."
            except AssertionError:
                raise
            # turn into array for easy broadcasting
            observables_weights = np.array(observables_weights)
            # normalise weight values such that sum to 1
            self.observables_weights = observables_weights / observables_weights.sum()

        else:
            self.observables_weights = None

        # store useful probabilities and inferences when calculated
        self.gamma = None
        self.xi = None
        self.Z0_probs = None
        self.Z1_probs = None
        self.Z_inferences = None

    def __repr__(self) -> str:

        """
        Represent the HMM in terms of its parameters.
        """

        return f"{self.N} hidden states; {self.T} time points; initial state probability P(Z0) = {self.pi[0]}; transition probs: Z0 -> Z0 = {self.A[0,0].round(decimals=3)} and Z1 -> Z1 = {self.A[1,1].round(decimals=3)}."

    def B_oi(self, B: np.ndarray, zi: int, o: int) -> np.float32:

        """
        Finds the emission probability, b_{i}(o), of observing a given variable value, o, in a given state.

        Parameters
        ----------
        B : np.ndarray[float]
            Emission probabilities matrix, of dims (N,K,2), where N=number of hidden state; K=number of different possible observations;
            and the final dimension of 2 refers to the two leaves - the first containing variable values, and the second concomitant emission
            probabilities.
        zi : int
            Hidden state.
        o : int
            Value of the observed variable.

        Returns
        -------
        b_oi : float
            Emission probability extracted from the emission probability vector.
        """

        # find index location of the observed value within the first leaf
        idx = np.where(B[zi, :, 0] == o)[0][0]

        # extract concomitant emission probability stored in the next leaf
        b_oi = B[zi, idx, 1]

        return b_oi

    def forwards_compute(self, O: list[int], B: np.ndarray) -> np.ndarray:

        """
        Implements the forward pass of the forwards-backwards algorithm, whilst implementing the "log-sum-exp" trick to avoid underflow, and therefore returning log(alpha), where

        - alpha_{i}(t) = P(o_{1:t}, Z_{t} | theta), and where theta = (A,B,pi).

        This essentially describes the probability of seeing forward obervations up to time t (that is o_{1}, o_{2}, ...,o_{t}), and being in state i at time t, for t=1:T, given the current
        HMM parameters theta.

        Parameters
        ----------
        O : list[int]
            Observed variable.
        B : np.ndarray[float]
            Emission probabilities matrix, of dims (N,K), where K=number of different possible observations.

        Returns
        -------
        alpha_log : np.ndarray[float]
            Array of log(alpha values), dims = (N,T), where T=number of time points.
        """

        # disable Numpy's "RuntimeWarning: divide by zero encountered in log" error message
        with np.errstate(divide='ignore'):

            # instantiate alpha_log
            alpha_log = np.zeros((self.N, self.T), dtype=np.float32)

            # initalisation
            for i in self.Z:
                alpha_log[i, 0] = np.log(self.pi[i]) + np.log(self.B_oi(B, i, O[0]))

            # recursion with log-sum-exp trick
            alpha_paths = np.zeros(
                (self.N, 1), dtype=np.float32
            )  # instantiate array once outside of loop (then overwrite iteratively)

            for t in range(self.T - 1):
                for i in self.Z:

                    # perform sum over j=1:N, whilst implementing log-sum-exp trick to avoid underflow
                    for j in self.Z:
                        alpha_paths[j, 0] = alpha_log[j, t] + np.log(self.A[j, i])

                    # extract a constant for the log-sum-exp trick (take max alpha path value)
                    c = alpha_paths.max()

                    # # shift values by max val (i.e. max val now = 0) and exponentiate
                    alpha_paths = np.exp(alpha_paths - c)

                    # calculate log of alpha
                    alpha_log[i, t + 1] = np.log(self.B_oi(B, i, O[t + 1])) + (
                        c + np.log(np.sum(alpha_paths))
                    )

        return alpha_log

    def backwards_compute(self, O: list[int], B: np.ndarray) -> np.ndarray:

        """
        Implements the backward pass of the forwards-backwards algorithm, whilst implementing the "log-sum-exp" trick to avoid underflow, and therefore returning log(beta), where

        - beta_{i}(t) = P(o_{t+1:T} | Z_{t}=i, theta), and where theta = (A,B,pi).

        This essentially describes the probability of observing the partial sequence from o_{t+1}:o_{T}, given the hidden state i at time t (Z_{t}=i), and the current HMM parameters theta.

        Parameters
        ----------
        O : list[int]
            Observed variable.
        B : np.ndarray[float]
            Emission probabilities matrix, of dims (N,K), where K=number of different possible observations.

        Returns
        -------
        beta_log : np.ndarray[float]
            Array of log(beta) values, dims = (N,T), where T=number of time points.
        """

        # disable Numpy's "RuntimeWarning: divide by zero encountered in log" error message
        with np.errstate(divide='ignore'):

            # instantiate beta_log
            beta_log = np.zeros((self.N, self.T), dtype=np.float32)

            # initalisation
            for i in self.Z:
                beta_log[i, self.T - 1] = np.log(1.0)

            # recursion, iterating backwards
            beta_paths = np.zeros(
                (self.N, 1), dtype=np.float32
            )  # instantiate array once outside of loop (then overwrite iteratively)

            for t in range(self.T - 2, -1, -1):  # akin to T-2:-1:0 inclusive
                for i in self.Z:

                    for j in self.Z:
                        beta_paths[j, 0] = (
                            beta_log[j, t + 1]
                            + np.log(self.A[i, j])
                            + np.log(self.B_oi(B, j, O[t + 1]))
                        )

                    # extract a constant for the log-sum-exp trick
                    c = beta_paths.max()

                    # shift values by max val (i.e. max val now = 0) and exponentiate
                    beta_paths = np.exp(beta_paths - c)

                    # calculate log of beta
                    beta_log[i, t] = c + np.log(np.sum(beta_paths))

        return beta_log

    def gamma_compute(self, alpha_log: np.ndarray, beta_log: np.ndarray) -> np.ndarray:

        """
        Uses alpha_log and beta_log to compute the array of tempory variables, log(gamma), where gamma describes the probability of being in state i at time t given the
        observed sequence O and HMM parameters theta. Implements the log-sum-exp trick to avoid underflow.

        - gamma_{i}(t) = P(Z_{t}=i | O, theta) = P(Z_{t}=i, O | theta) / P(O | theta)

        Parameters
        ----------
        alpha_log : np.ndarray[float]
            Array of log(alpha) values; expected dims = (N,T), where N=number of states and T=number of time points.
        beta_log : np.ndarray[float]
            Array of log(beta) values; expected dims = (N,T).

        Returns
        -------
        gamma : np.ndarray[float]
            P(Z_{t}=i | O, theta); dims=(N,T).
        Z_inference : np.ndarray[int]
            Inferred hidden state sequence.
        """

        # disable Numpy's "RuntimeWarning: divide by zero encountered in log" error message
        with np.errstate(divide='ignore'):

            # instantiate gamma
            gamma_log = np.zeros((self.N, self.T), dtype=np.float32)

            # instantiate ais array, where a_{i} = log(alpha_{i}(t)) + log(beta_{i}(t))
            ais = np.zeros((self.N, 1), dtype=np.float32)

            # iterate through series
            for t in range(self.T):

                # extract the log(alpha_{i}(t)) + log(beta_{i}(t)) terms
                for i in self.Z:
                    ais[i, 0] = alpha_log[i, t] + beta_log[i, t]

                # extract constant for log-sum-exp operation
                c = ais.max()

                # make copy for denominators
                denominator_log = ais.copy()

                # shift vals and exponentiate
                denominator_ais_shift = np.exp(denominator_log - c)
                for i in self.Z:
                    numerator_ai_shift = np.exp(ais[i] - c)

                    # calculate and store log(gamma_{i}(t)) values
                    gamma_log[i, t] = (c + np.log(numerator_ai_shift)) - (
                        c + np.log(np.sum(denominator_ais_shift))
                    )

            # exponentiate final results
            gamma = np.exp(gamma_log)

            # infer the Z sequence
            Z_inference = np.zeros((1, self.T), dtype=np.int32)

            for t in range(self.T):
                Z_inference[0, t] = gamma[:, t].argmax()

        return gamma, gamma_log, Z_inference

    def xi_compute(
        self,
        alpha_log: np.ndarray,
        beta_log: np.ndarray,
        O: list[int],
        B: np.ndarray,
    ) -> np.ndarray:

        """
        Uses alpha_log, beta_log, transition and emission probabilities to compute the probability of being in states i at j at times t and t+1, respectively, given the observed
        sequence O and HMM parameters theta. That is:

        - xi_{ij}(t) = P(Z_{t}=i, Z_{t+1}=j| O, theta) = P(Z_{t}=i, Z_{t+1}=j, O | theta) / P(O | theta)

        Therefore, xi will be of len=T-1 (as looking at transitions from t -> t+1), and at each t we will have a matrix of transition probabilities, leading to dims of (N, N, T-1).

        This function implements the "log-sum-exp" trick to circumvent the problem of underflow, here encountered when dealing with vanishingly small probabilities.

        Parameters
        ----------
        alpha_log : np.ndarray[float]
            Array of log(alpha) values; expected dims = (N,T), where N=number of states and T=number of time points.
        beta_log : np.ndarray[float]
            Array of log(beta) values; expected dims = (N,T).
        O : list[int]
            Observed variable (elements expected to be ints).
        B : np.ndarray[float]
            Emission probabilities matrix, of dims (N,K), where K=number of different possible observations.

        Returns
        -------
        xi : np.ndarray[float]
            P(Z_{t}=i, Z_{t+1}=j| O, theta); dims=(N, N, T-1).
        xi_log : np.ndarray[float]
            Natural logarithm of xi; dims=(N, N, T-1).
        """

        # disable Numpy's "RuntimeWarning: divide by zero encountered in log" error message
        with np.errstate(divide='ignore'):

            # instantiate xi
            xi_log = np.zeros((self.N, self.N, self.T - 1), dtype=np.float32)

            # instantiate array
            numerators_log = np.zeros((self.N, self.N), dtype=np.float32)

            # iterate through series
            for t in range(self.T - 1):

                # extract the NxN array: alpha * transition_prob * beta * emission_prob terms
                for k in self.Z:
                    for w in self.Z:
                        numerators_log[k, w] = (
                            alpha_log[k, t]
                            + np.log(self.A[k, w])
                            + beta_log[w, t + 1]
                            + np.log(self.B_oi(B, w, O[t + 1]))
                        )

                # extract a constant for log-sum-exp
                c = numerators_log.max()

                # keep copy of denominators as log values
                denominator_log = numerators_log.copy()

                # shift and exponentiate the copy
                denominator_shift_exp = np.exp(denominator_log - c)

                # now iterate through the NxN elements of the denominator array, extract the element, shift the value and exponentiate, compue xi, and shift back
                for i in self.Z:
                    for j in self.Z:
                        # extract values, shift and exponentiate
                        numerator_ij_shift = np.exp(numerators_log[i, j] - c)

                        # calculate xi_log and store for each time step
                        xi_log[i, j, t] = (c + np.log(numerator_ij_shift)) - (
                            c + np.log(np.sum(denominator_shift_exp))
                        )

        # exponentiate results
        xi = np.exp(xi_log)

        return xi, xi_log

    def forwards_backwards_algorithm(self) -> tuple[np.ndarray]:

        """
        Implements the forward-backwards algorithm to return gamma and xi; log(alpha) and log(beta) are stored as instance attributes.

        Returns
        -------
        gamma : np.ndarray[float]
            P(Z_{t}=i | O, theta); dims=(N,T, R).
        xi : np.ndarray[float]
            P(Z_{t}=i, Z_{t+1}=j| O, theta); dims=(N, N, T-1, R).
        """

        # instantiate arrays for gamma and xi, including the R dimension
        alpha_log = np.zeros((self.N, self.T, self.R), dtype=np.float32)
        beta_log = np.zeros((self.N, self.T, self.R), dtype=np.float32)
        gamma = np.zeros((self.N, self.T, self.R), dtype=np.float32)
        xi = np.zeros((self.N, self.N, self.T - 1, self.R), dtype=np.float32)

        # iterate through all observables (with appropriate emission probs matrix)
        for r, (O_r, B_r) in enumerate(zip(self.O_list, self.B_list)):

            # forward pass
            alpha_log_r = self.forwards_compute(O_r, B_r)

            # backward pass
            beta_log_r = self.backwards_compute(O_r, B_r)

            # compute gamma
            gamma_r, _, _ = self.gamma_compute(alpha_log_r, beta_log_r)

            # compute xi
            xi_r, _ = self.xi_compute(
                alpha_log_r,
                beta_log_r,
                O_r,
                B_r,
            )

            # store results from each observable
            alpha_log[:, :, r] = alpha_log_r
            beta_log[:, :, r] = beta_log_r
            gamma[:, :, r] = gamma_r
            xi[:, :, :, r] = xi_r

        # store as instance attributes (previously set as None)
        self.alpha_log, self.beta_log, self.gamma, self.xi = (
            alpha_log,
            beta_log,
            gamma,
            xi,
        )

        return gamma, xi
    
    def log_likelihood(self):

        """
        Computes the log-likelihood of alpha:
        
        - alpha_{i}(t) = P(o_{1:t}, Z_{t} | theta), and where theta = (A,B,pi).

        This is simply the sum of the log of alpha.

        - log-likelihood = sum(log(alpha)).

        This enables the implementation of a simple "early stopping" approach to the Baum-Welch expectation maximisation iterative optimisation.
        """

        # log_likelihood_alpha = np.sum(alpha_log)
        log_likelihood_alpha = np.average(self.alpha_log[:,-1,:])

        return log_likelihood_alpha

    def B_update(self, gamma: np.ndarray, B: np.ndarray, O: list[int]):

        """
        Computes the expected emission probabilities, given gamma and the sequence of observed variables, O.

        Parameters
        ----------
        gamma : np.ndarray[float]
            P(Z_{t}=i | O, theta); dims=(N,T).
        B : np.ndarray[float]
            Emission probabilities matrix, of dims (N,K), where K=number of different possible observations.
        O : list[int]
            List of observed variables (elements expected to be ints).

        Returns
        -------
        B : np.ndarray[float]
            Updated emission probabilities matrix, of dims (N,K), where K=number of different possible observations.
        """

        # instantiate new transmission probabilities matrix
        B_update = np.zeros((B.shape), dtype=np.float32)

        # take the sums for gamma in all N states along the t axis
        gamma_i_sum = gamma.sum(axis=1).reshape(self.N, 1)

        # iterate through hidden states, potential observed values (oi), and time, to find expected output observations whilst in state i
        for i in self.Z:
            for oi in range(B.shape[1]):

                # instantiate cumulative sum for numerator
                bi_cum_sum = 0
                for t in range(self.T):

                    # if observation seen, in that state, at time t, add the gamma value
                    if O[t] == oi:
                        bi_cum_sum += gamma[i, t]

                # update B
                B_update[i, oi, 1] = bi_cum_sum / gamma_i_sum[i, 0]

        return B_update

    def pi_multiple_obs_update(self, gamma: np.ndarray) -> list[float]:

        """
        Extracts the updated inital state probabilities, pi, from gamma.

        Parameters
        ----------
        gamma : np.ndarray[float]
            P(Z_{t}=i | O, theta); dims=(N,T,R), where R is the number of observable variables.

        Returns
        -------
        list[float]
            The updated initial state probabilities, pi, of len=N.
        """

        # initial state probs, averaged from all observables, to update pi (axis=-1 as last dimension the R dimension)
        if self.observables_weights is not None:
            return list(
                np.average(gamma[:, 0, :], axis=-1, weights=self.observables_weights)
            )
        else:
            return gamma[:, 0, :].mean(axis=-1).tolist()

    def A_multiple_obs_update(self, gamma: np.ndarray, xi: np.ndarray) -> np.ndarray:

        """
        Computes the expected transmission probabilities.

        Parameters
        ----------
        gamma : np.ndarray[float]
            P(Z_{t}=i | O, theta); dims=(N, T, R).
        xi : np.ndarray[float]
            P(Z_{t}=i, Z_{t+1}=j| O, theta); dims=(N, N, T-1, R).

        Returns
        -------
        A_update : np.ndarray[float]
            Updated transmission probabilities matrix, of dims (N,N), where N=number of hidden states.
        """

        # instantiate array
        A_update = np.zeros((self.N, self.N), dtype=np.float32)

        # sum arrays along T axis and then R axis (which will be final axis=-1)
        gamma_sum = gamma[:, :-1, :].sum(axis=(1, -1)).reshape(self.N, 1)
        xi_sum = xi.sum(axis=(2, -1))

        for i in self.Z:
            for j in self.Z:
                A_update[i, j] = xi_sum[i, j] / gamma_sum[i, 0]

        return A_update

    def baumwelch_expectationMaximisation(
        self,
        iter: int = 1,
        update_pi: bool = True,
        update_A: bool = True,
        update_B: bool = True,
        early_stopping: bool=True,
        log_likelihood_p_delta: float=0.005,
        rolling_deltas: int=3,

    ) -> tuple[np.ndarray]:

        """
        Implements the Baum-Welch algorithm, i.e., the forwards-backwards followed by HMM parameter updates (theta).

        Parameters
        ----------
        iter : int, default=1
            Iterations of expectation maximisation for the Baum-Welch algorithm.
        update_pi : bool, default=True
            Whether to update the initial state probabilities.
        update_A : bool, default=True
            Whether to update the transition state probabilities.
        update_B : bool, default=True
            Whether to update the emission probabilities.
        """

        # instantiate
        log_likelihood_deltas = []

        for i in range(iter):

            # forwards-backwards algorithm across all observables
            gamma, xi = self.forwards_backwards_algorithm()

            # update initial state probs, pi
            if update_pi:
                self.pi = self.pi_multiple_obs_update(gamma)

            # update transition probs
            if update_A:
                self.A = self.A_multiple_obs_update(gamma, xi)

            # update emission probs
            if update_B:
                self.B_list = [
                    self.B_update(gamma[:, :, r], B, O)
                    for r, (B, O) in enumerate(zip(self.B_list, self.O_list))
                ]

            # if implementing early stopping
            if early_stopping:

                # compute log-likelihood
                log_likelihood_alpha = self.log_likelihood()

                # assign log_likelihood_alpha_prev for first iteration
                if i == 0:
                    log_likelihood_alpha_prev = log_likelihood_alpha

                # calculate difference and store values
                log_likelihood_delta = abs(log_likelihood_alpha - log_likelihood_alpha_prev)
                log_likelihood_deltas.append(log_likelihood_delta)

                # look at rolling average of differences
                if i>=rolling_deltas:
                    deltas = np.array(log_likelihood_deltas[i:i+rolling_deltas], dtype=np.float64)
                    rolling_mean = deltas.mean()

                    # if threshold larger than rolled averages, stop
                    if rolling_mean<log_likelihood_p_delta:
                        print(f"Early stopping converged on iteration {i+1}.")
                        break
                    
                # assign log_likelihood_alpha_prev for next iteration
                log_likelihood_alpha_prev = log_likelihood_alpha

        # store final gamma and xi arrays as attributes
        self.gamma = gamma
        self.xi = xi

        return self
    
    def Z_state_probs_inference(self, **kwargs : dict) -> tuple[np.ndarray]:

        """
        Given gamma array of dims (N, T, R), where gamma is defined as:

        - gamma_{i}(t) = P(Z_{t}=i | O, theta) = P(Z_{t}=i, O | theta) / P(O | theta)

        Average the results along the R axis (i.e., across all observables), or by the weights provided, and return the Z inference probabilities and states.

        Parameters
        ----------
        kwargs : dict
            Dictionary of keyword arguments: weights : list[int | float].

        Returns
        -------
        Z0_prob : np.ndarray[float]
            P(Z_{i} when i=0 | O_{1}, ..., O_{R}, theta); expected dims: (1,T).
        Z1_prob : np.ndarray[float]
            P(Z_{i} when i=1 | O_{1}, ..., O_{R}, theta); expected dims: (1,T).
        Z_inference : np.ndarray[int]
            Inferred hidden state sequence; expected dims: (1, T).
        gamma_meanR : np.ndarray[float]
            P(Z_{i} where i=row | O_{1}, ..., O_{R}, theta); expected dims: (N,T).
        """

        # calculate P(Z_{i} when i=0 | O_{1}, ..., O_{R}, theta) and P(Z_{i} when i=1 | O_{1}, ..., O_{R}, theta)
        # eithre weight the mean by provided values, or if none provided, take mean across the R axis (always the final axis) - i.e., treat all variables as evenly weighted
        if 'weights' in kwargs:

            # quality checks
            try:
                assert isinstance(kwargs['weights'], list), "Keyword argument `weights` in z_state_probs_inference must be a lits of len=number hidden states."
                assert (self.gamma.shape[0] == len(kwargs['weights'])), "Number of weights does not match number of hidden states."
                assert all(isinstance(weight, (int, float)) for weight in kwargs['weights']), "Elements in `weights` for z_state_probs_inference must be numeric."
            except AssertionError:
                raise

            # assuming pased quality checks, average using the weights provided
            gamma_meanR = np.average(self.gamma, axis=-1, weights=kwargs['weights'])

        else: # using weights provided at init, if provided

            if self.observables_weights is not None:
                gamma_meanR = np.average(self.gamma, axis=-1, weights=self.observables_weights)
            else:
                gamma_meanR = self.gamma.mean(axis=-1)

        # P(Z_{i} when i=0 or i=1 | O_{1}, ..., O_{R}, theta)
        Z0_prob = gamma_meanR[0, :].reshape(1, self.T)
        Z1_prob = gamma_meanR[1, :].reshape(1, self.T)

        # extract the inferred state
        Z_inference = np.zeros((1, self.T), dtype=np.int32)
        for t in range(self.T):
            Z_inference[0, t] = gamma_meanR[:, t].argmax()

        return Z0_prob, Z1_prob, Z_inference, gamma_meanR