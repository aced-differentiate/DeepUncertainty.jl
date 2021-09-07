module DeepUncertainty

# Export layers 
export MCLayer, MCDense, MCConv
export mean_loglikelihood, brier_score, ExpectedCalibrationError, prediction_metrics

include("metrics.jl")
include("layers/mclayers.jl")

end
