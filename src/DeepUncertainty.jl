module DeepUncertainty

# Export layers 
export MCLayer, MCDense, MCConv
export DenseBatchEnsemble, ConvBatchEnsemble
export mean_loglikelihood, brier_score, ExpectedCalibrationError, prediction_metrics

include("metrics.jl")
include("layers/mclayers.jl")
include("layers/BatchEnsemble/dense.jl")
include("layers/BatchEnsemble/conv.jl")

end
