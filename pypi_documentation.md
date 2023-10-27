# hmmBaumWelch

Implements the Baum-Welch expectation maximisation algorithm to update the parameters and perform inference on hidden Markov models (HMM) with observed time-series data (for one or multiple variables). In brief, this package maximises the local probabilty of observing the given sequence of variables by updating the HMM initial state, prior, and transition state probabilities. It then performs inference to give the hidden state sequence as well as rendering the probability of a given hidden state at each time point.

The package is designed to work with SciPy's probability distribution functions to simplify the defining of the Bayesian priors for each variable and state; and a simple form of "early stopping" has been implemented to optimise performance.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install hmmBaumWelch.

```bash
pip install hmmBaumWelch
```

## Usage

```python
from hmmBaumWelch import BaumWelch
```

For examples of package implementation, see the `examples` directory.

When instantiating the BaumWelch object, several arguments are expected:

1. N: the number of hidden states within the system. E.g., if N=2, Zi will be in set {0,1}.
2. O_list: a list of lists representing each observed, numeric variable. These can be continuous or discreet, but must contain real numbers. They must also be of equal lengths.
3. pi: the starting probabilities (i.e. at t=0) for all hidden states. This is a list of length=N.
4. A: the transition state matrix: a square numpy array, in which each dimension is equal to the number of states N. Each position represents the probability of transitioning between states, i.e., A(0,0) = P(Zi=0,t=t, Zi=0,t=t+1) - the probability that a hidden state will remain in Zi=0 from one time point to the next. Therefore each row of A must sum to one.
5. B_list: a list of prior probability distributions for each variable and all hidden states. These probability distributions can be defined straight from the Scipy package.

An example might be:

`HMM = BaumWelch(N=2, O_list=O_list, pi=[0.5, 0.5], A=np.array([[0.8, 0.2], [0.2, 0.8]]), B_list=B_list)`

After instantiating the object, expectation maximisation is performed to fit the hidden Markov model. For example:

`HMM.baumwelch_expectationMaximisation(iter=50, update_pi=True, update_A=True, update_B=False, early_stopping=True)`

The arguments state that there will be a maximum of 50 iterations of expectation maximisation (stopping early if a good fit found); the starting probabilities, pi, and transition state probability matrix, A, will be updated also, but the priors will remain static. The results can then be extracted with the `Z_state_probs_inference` method, e.g.:

`_, _, Z_inference, gamma_meanR = HMM.Z_state_probs_inference()`

`Z_inference` is an array of integers, stating the inferred hidden states at each time step; and gamma_meanR is the probability of each hidden state at each time step, given the final HMM parameters. If the optional `observables_weights` parameter was entered at instantiation, each variable will have a concomitant weighting in the parameter updates and final inference.

## Theory

For a summary of the Baum-Welch algorithm, see the [Wikipedia page](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm); and for formal mathematical notation, see the project readme.

## Contributors

George Stanley.

## License

[GNU GENERAL PUBLIC LICENSE Version 2](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
