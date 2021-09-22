# DeepUncertainty

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://DwaraknathT.github.io/DeepUncertainty.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://DwaraknathT.github.io/DeepUncertainty.jl/dev)
[![Build Status](https://github.com/DwaraknathT/DeepUncertainty.jl/workflows/CI/badge.svg)](https://github.com/aced-differentiate/DeepUncertainty.jl/actions)
[![Coverage](https://codecov.io/gh/DwaraknathT/DeepUncertainty.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/DwaraknathT/DeepUncertainty.jl)


Tools for uncertianty estimation in Deep Learning models.
```
] add DeepUncertainty
```

The package is structure the following way 
```
DeepUncertainty/
├── docs
│   └── src
├── examples                # Examples using the package 
├── src
│   └── layers
│       ├── BatchEnsemble   # BatchEnsemble layers 
│       └── Variational     # Variational inference layers 
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