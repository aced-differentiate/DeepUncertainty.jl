# DeepUncertainty

Documentation for [DeepUncertainty](https://github.com/DwaraknathT/DeepUncertainty.jl).

DeepUncertainty implements techniques generally used to qualtify uncertainty in neural networks. It implements a variety of methods such [Monte Carlo Dropout](https://arxiv.org/abs/1506.02142), [BatchEnsemble](https://arxiv.org/abs/2002.06715), [Bayesian BatchEnsemble](https://arxiv.org/abs/2005.07186), [Bayesian Neural Networks](https://arxiv.org/abs/1505.05424). The goal is to have drop-in replacements for Dense and Conv layers to immediatly convert deterministic networks to Bayesian networks, and also provide examples to help convert custom layers to Bayesian.

This package is in development as part of the [ACED project](https://www.cmu.edu/aced/), funded by ARPA-E DIFFERENTIATE and coordinated by [Carnegie Mellon University](https://www.cmu.edu/), in collaboration with [Julia Computing](https://juliacomputing.com/), [Citrine Informatics](https://citrine.io/), and [MIT](https://web.mit.edu/).

