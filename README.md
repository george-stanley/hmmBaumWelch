# hmmBaumWelch

Implements the Baum-Welch expectation maximisation algorithm to update the parameters and perform inference on hidden Markov models (HMM) with observed time-series data (for one or multiple variables). In brief, this package maximises the local probabilty of observing the given sequence of variables by updating the HMM initial state, prior, and transition state probabilities. It then performs inference to give the hidden state sequence as well as rendering the probability of a given hidden state at each time point.

The package is designed to work with SciPy's probability distribution functions to simplify the defining of the Bayesian priors for each variable and state; and a simple form of "early stopping" has been implemented to optimise performance.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install hmmBaumWelch.

```bash
pip install -i https://test.pypi.org/simple/ hmmBaumWelch
```

## Usage

```python
from hmmBaumWelch import BaumWelch
```

## Theory

For a summary of the Baum-Welch algorithm, see the [Wikipedia page.](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm)

### Formal notation

- $`O_{r}`$ is a discrete observed variable in the set $`\{O_{r}, ..., O_{R}\}`$, where $`R`$ is the number of observed variables;
- $`T`$ is the total number of time points in the sequence of observed variables, where $`t \in \{1, 2, ..., T\}`$;
- $`Z_{t}=i`$ is the hidden state at time, t.

The hidden Markov model is described by 3 parameters: the transition state probabilities, $`A`$; prior probabilities, $`B`$; and initial state probabilities, $`\pi`$; each defined as follows:

- **Transition state probabilities:** $`A = \{a_{ij}\} = P(Z_{t}=j|Z_{t-1}=i)`$
- **Initial state probabilities:** $`\pi_{i} = P(Z_{1}=i)`$

And if the observed variable $`O_{r}`$ can take one of $`K`$ possible values, and assuming an observation — given a hidden state — is time independent, we can say:

- $`b_{r,i}(o_{r,k}) = P(o_{r,k}|Z_{t}=i)`$

That is the probability of seeing a given value $`o_{r,k}`$, of observed variable $`O_{r}`$, in state $`Z_{i}`$.

Accounting for all possible values of $`O_{r,t}`$ and $`Z_{t}`$, we can represent the prior probabilities as a matrix of dimensions $`N \times K`$, where $`N`$ is the number of possible hidden states. So we represent the prior probabilities matrix, for an observable $`O_{r}`$, and a hidden state $`Z=i`$ as:

- **Initial state probabilities:** $`B_{r,i} = \{b_{r,i}(o_{r,k})\}`$

Grouping these together, we define the hidden Markov model as this collection of parameters:

- **HMM parameters:** $`\theta=(A,B,\pi)`$

The goal of the Baum-Welch expectation maximisation is to iteratively update these parameters to find the local maximal possibility of observing the given sequence of variables, that is:

- $`\theta^{*} = argmax_{\theta}P(O_{r}, ...,O_{R}|\theta)`$.

### Forwards-backwards algorithm

The algorithm begins with the two recursive *forwards* and *backwards* passes through the seqeunce of observed variables, computing two posterior marginal quantities for all hidden states:

- **Forwards:**  $`\alpha_{r, i}(t) = P(O_{r, 1:t}, Z_{t}=i| \theta)`$
- **Backwards:** $`\beta_{r, i}(t) = P(O_{r, t+1:T} | Z_{t}=i, \theta)`$

**NB:** As these quantities include the probability of observing sequences up to a given time point — $`o_{r, 1:t}`$ or $`o_{r, t+1:T}`$ — the two series converge exponentially to zero. Therefore, to avoid underflow, both $`\alpha`$ and $`\beta`$ are represented as natural logarithms, and for all computations the "log-sum-exp" trick is used.

#### Forward pass

For $`Z_{t} = i`$ and $`O_{r}`$:

1. $`\alpha_{r, i}(1) = \pi_{i}b_{r, i}(o_{r, 1})`$

2. $`\alpha_{r, i}(t+1) = b_{r, i}(o_{r, t+1}) \sum_{j=1}^{N}\alpha_{r, j}(t)a_{ji}`$

#### Backward pass

For $`Z_{t} = i`$ and $`O_{r}`$:

3. $`\beta_{r, i}(T) = 1`$

4. $`\beta_{r, i}(t) = \sum_{j=1}^{N}\beta_{r, j}(t+1)a_{ij}b_{r, j}(o_{r, t+1})`$

### Update

Using $`\alpha`$ and $`\beta`$ we can now compute two more conditional probabilities, for given $`O_{r}`$ and $`Z_{i}`$:

5. $`\gamma_{r, i}(t) = P(Z_{t}=i | O_{r}, \theta) = \frac{P(Z_{t}=i, O_{r} | \theta)}{P(O_{r} | \theta)} = \frac{\alpha_{r, i}(t)\beta_{r, i}(t)}{\sum_{j=1}^{N}\alpha_{r, j}(t)\beta_{r, j}(t)}`$

6. $`\xi_{r, ij}(t) = P(Z_{t}=i, Z_{t+1}=j | O_{r}, \theta) = \frac{P(Z_{t}=i, Z_{t+1}=j, O_{r} | \theta)}{P(O_{r} | \theta)} = \frac{\alpha_{r, i}(t) a_{ij} \beta_{r, j}(t+1) b_{r, j}(o_{r, t+1}) }{\sum_{k=1}^{N} \sum_{w=1}^{N} \alpha_{r, k}(t) a_{kw} \beta_{r, w}(t+1) b_{r, w}(o_{r, t+1})}`$

**NB:** The "log-sum-exp" trick is also employed in the computational steps 5 and 6 for the same reasons as above.

These quantities are then used to update the HMM parameters $`\theta`$, for a given $`O_{r}`$:

7. $`\pi_{r, i}^{*} = \gamma_{r, i}(1)`$

8. $`a_{r, ij}^{*} \frac{\sum_{t=1}^{T-1} \xi_{r, ij}(t)}{\sum_{t=1}^{T-1} \gamma_{r, i}(t)}`$

9. $`b_{r, i}^{*}(v_{k}) = \frac{\sum_{t=1}^{t} 1_{o_{r, t}=v_{k}} \gamma_{r, i}(t)}{\sum_{t=1}^{T}\gamma_{r, i}(t)}`$

where $`1_{o_{r, t}=v_{k}} = 1`$ if $`o_{r, t}=v_{k}`$, othwerwise 0.

**NB:** These update parameters are relative to the observed variable. If there are multiple observed variables (*i.e.* $`R > 1`$), mean values are taken for $`\pi_{i}^{*}`$ and $`a_{ij}^{*}`$. Updates to the priors remain relative to each observed variable.

That is:

10. $`\pi_{i}^{*} = \frac{\sum_{r=1}^{R} \gamma_{r, i}(1)}{R}`$

11. $`a_{ij}^{*} = \frac{\sum_{r=1}^{R} \sum_{t=1}^{T-1} \xi_{r, ij}(t)}{\sum_{r=1}^{R} \sum_{t=1}^{T-1} \gamma_{r, i}(t)}`$

By iteratively repeating steps 1 through 11 we update $`\theta`$ to find the local maximum. And by reviewing $`\gamma`$ after convergence, we can infer the quantity:

- $`P(Z_{t}=i | O_{r}, \theta)`$

for each and all observed variables.

## Contributors

George Stanley.

## License

[GNU GENERAL PUBLIC LICENSE Version 2](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
