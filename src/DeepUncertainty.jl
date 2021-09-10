module DeepUncertainty

using Random
using DistributionsAD

using Flux
using Flux: @functor, create_bias, params
using Flux: glorot_uniform, glorot_normal

export KLDivergence
export TrainableDistribution
# Export layers 
export MCLayer, MCDense, MCConv
export DenseBatchEnsemble, ConvBatchEnsemble
export mean_loglikelihood, brier_score, ExpectedCalibrationError, prediction_metrics

include("metrics.jl")
include("initializers.jl")
include("regularizers.jl")

# Layers 
include("layers/mclayers.jl")
include("layers/BatchEnsemble/dense.jl")
include("layers/BatchEnsemble/conv.jl")

end
