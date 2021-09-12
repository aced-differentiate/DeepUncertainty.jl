module DeepUncertainty

using Random
using DistributionsAD

using Flux
using Flux: @functor, create_bias, params
using Flux: glorot_uniform, glorot_normal

export KLDivergence
export TrainableDistribution, AbstractTrainableDist
# Export layers 
export MCLayer, MCDense, MCConv
export VariationalConv, VariationalDense
export DenseBatchEnsemble, ConvBatchEnsemble
export ExpectedCalibrationError, prediction_metrics
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

end
