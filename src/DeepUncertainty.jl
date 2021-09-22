module DeepUncertainty

using Random
using DistributionsAD

using Flux
using Flux: @functor, create_bias, params
using Flux: glorot_uniform, glorot_normal

export TrainableMvNormal
export normal_kl_divergence, scale_mixture_kl_divergence

# Export layers 
export MCLayer, MCDense, MCConv
export VariationalConv, VariationalDense
export DenseBE, ConvBE
export VariationalConvBE, VariationalDenseBE
export expected_calibration_error, prediction_metrics
export mean_loglikelihood, brier_score, calculate_entropy

include("metrics.jl")
include("initializers.jl")
include("regularizers.jl")

# Layers 
include("layers/mclayers.jl")
include("layers/BatchEnsemble/dense.jl")
include("layers/BatchEnsemble/conv.jl")
include("layers/Variational/conv.jl")
include("layers/Variational/dense.jl")
include("layers/BayesianBE/conv.jl")
include("layers/BayesianBE/dense.jl")

end
