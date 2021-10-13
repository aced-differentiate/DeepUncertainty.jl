# DeepUncertainty

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://aced-differentiate.github.io/DeepUncertainty.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://aced-differentiate.github.io/DeepUncertainty.jl/dev)
[![Build Status](https://github.com/aced-differentiate/DeepUncertainty.jl/workflows/CI/badge.svg)](https://github.com/aced-differentiate/DeepUncertainty.jl/actions)
[![codecov](https://codecov.io/gh/aced-differentiate/DeepUncertainty.jl/branch/main/graph/badge.svg?token=9XDVJ3TOE3)](https://codecov.io/gh/aced-differentiate/DeepUncertainty.jl)


Tools for uncertianty estimation in Deep Learning models. Use the below command in REPL to install. 
```
] add DeepUncertainty
```

## Basics 

Neural Networks are usually trained to minimize a loss function that approximates model performance on any given task. The weights (parameters) of the network are estimated using first-order gradient based optimization algorithms that usually result in a point estimates. 

We can also take take a Bayesian view point and place a distribution on each weight instead of interpreting it as a point estimate. We can calculate the uncertianty in our estimates of the parameters, since we optimize the distribution parameters insteaf of point estimates directly. 

## Current Inventory
* [MC Dropout](https://arxiv.org/abs/1506.02142) 
* [BatchEnsemble](https://arxiv.org/abs/2002.06715)
* [Variational Inference](https://arxiv.org/abs/1505.05424)
* [Bayesian BatchEnsemble](https://arxiv.org/abs/2005.07186)