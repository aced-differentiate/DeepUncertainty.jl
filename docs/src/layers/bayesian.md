# Bayesian Layers 

```@index
Pages=["bayesian.md"]
```

We follow the paper "[Weight Uncertainty in Neural Networks](https://arxiv.org/abs/1505.05424)" to implement Bayesian Layers. The paper proposes replacing point estimates of weights in the weight matric with distributions. So instead of optimizing for weights directly, we optimize the parameters of the distributions from which we sample weights. We sample a different set of weights at every forward pass. The predictive uncertainty is given by approximating the integral over the weight distribution using Monte Carlo sampling. 

The first component of creating Bayesian layers is trainable distributions, we use [DistributionsAD](https://github.com/TuringLang/DistributionsAD.jl) to backprop through distributions using [Zygote](https://github.com/FluxML/Zygote.jl), the flux autodiff framework. 
A trainable distribution should be a subtype of the type ```AbstractTrainableDist```

```@docs
DeepUncertainty.initializers
```