# MC Layers 

```@index
Pages=["mclayers.md"]
```

The goal of MC Layers is to facilitate converting any layer defined in Flux into its Bayesian counterpart. MC Dropout is a principled tecnique to estimate predictive uncertainty in neural networs. The paper [Dropout as a Bayesian Approximation](https://arxiv.org/abs/1506.02142) proves that a neural network with dropout applied before every layer is equivalent to a Deep Gaussain Process. We can apply dropout at test time to simulate functions draws in a Gaussian Process for approximate Variational Inference. 

In simple terms, a network with dropout applied before every layer has a different weight configuration in every forward pass (Note that we also apply dropout at test time, usually dropout is turned off during test time to ensemble the various weight samples implicitly). Each weight configuration can be considered an independent sample from the underlying weight distribution. The mean and variance of these sample predictions are the mean and variance of the predictive distribution. Higher the variance, more the uncertianty. 

```@docs
DeepUncertainty.layers.mclayers.MCLayer
```

We also implement MC versions of Conv and Dense layers as a drop-in replacement. 

```@docs
DeepUncertainty.layers.mclayers.MCDense
DeepUncertainty.layers.mclayers.MCConv 
```