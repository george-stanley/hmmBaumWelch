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

For examples of package implementation, see the `examples` directory.

## Theory

For a summary of the Baum-Welch algorithm, see the [Wikipedia page.](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm) And for formal mathematical notation, see the project readme.

## Contributors

George Stanley.

## License

[GNU GENERAL PUBLIC LICENSE Version 2](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
