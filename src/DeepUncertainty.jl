module DeepUncertainty

# Export layers 
export MCDense, MCConv
export mean_loglikelihood, brier_score, ExpectedCalibrationError

include("metrics.jl")
include("layers/mclayers.jl")

end
