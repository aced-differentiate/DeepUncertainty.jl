using Flux 
using Statistics
using CalibrationErrors 

function get_mean_std_dev(x::AbstractArray; dims=ndims(x))
    μ = mean(x, dims=dims)
    σ = std(x, dims=dims, corrected=false)
    return μ, σ
end

function ExpectedCalibrationError(preds, labels, num_bins=10;)
    preds = [x for x in eachcol(preds)]
    ece_estimator = ECE(UniformBinning(num_bins), (μ, y) -> kl_divergence(y, μ))
    ece = ece_estimator(preds, labels)
    return ece 
end 