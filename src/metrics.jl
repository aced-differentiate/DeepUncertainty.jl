using Flux 
using Statistics
using CalibrationErrors 

function get_mean_std_dev(x::AbstractArray; dims=ndims(x))
    μ = mean(x, dims=dims)
    σ = std(x, dims=dims, corrected=false)
    return μ, σ
end

function ExpectedCalibrationError(preds, labels, num_bins=10;)
    counts = zeros(num_bins)            # Total number of samples
    prob_sums = zeros(num_bins)         # Sum of all predicted probabilities for samples 
    correct_sums = zeros(num_bins)      # Total number of correct predictions 

    # # Predicted classes - the index with max prob
    # predicted_classes = onecold(preds)
    # # Predicted prob - the max prob predicted
    # predicted_probs = dropdims(maximum(preds, dims=1), dims=1) 
    # correct_predictions = (predicted_classes .== labels) 

    # # ECE 
    # println(size(predicted_probs))
    # println(size(labels))
    # println(predicted_probs)
    # println(labels) 
    # exit()
    ece_estimator = ECE(UniformBinning(num_bins), (μ, y) -> kl_divergence(y, μ))
    ece = ece_estimator(preds, labels)

    print(ece)
    exit()
    
end 