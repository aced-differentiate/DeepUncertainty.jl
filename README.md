# DeepUncertainty

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://DwaraknathT.github.io/DeepUncertainty.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://DwaraknathT.github.io/DeepUncertainty.jl/dev)
[![Build Status](https://github.com/aced-differentiate/DeepUncertainty.jl/workflows/CI/badge.svg)](https://github.com/aced-differentiate/DeepUncertainty.jl/actions)
[![codecov](https://codecov.io/gh/aced-differentiate/DeepUncertainty.jl/branch/main/graph/badge.svg?token=9XDVJ3TOE3)](https://codecov.io/gh/aced-differentiate/DeepUncertainty.jl)


Tools for uncertianty estimation in Deep Learning models.
```
] add DeepUncertainty
```

The package is structure the following way 
```
DeepUncertainty/
├── docs
│   └── src
├── examples
│   ├── Classification
│   │   ├── lenet
│   │   └── vgg
│   └── Regression
├── src
│   └── layers
│       ├── BatchEnsemble
│       ├── BayesianBE
│       └── Variational
└── test
    ├── cuda
    │   └── layers
    └── layers
```

## Current Inventory
* [MC Dropout](https://arxiv.org/abs/1506.02142) 
* [BatchEnsemble](https://arxiv.org/abs/2002.06715)
* [Variational Inference](https://arxiv.org/abs/1505.05424)
* [Bayesian BatchEnsemble](https://arxiv.org/abs/2005.07186)