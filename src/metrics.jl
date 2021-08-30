using Flux
using Statistics
using CalibrationErrors
using ReliabilityDiagrams

# TODO: Work on GPU friendly implementations of these metrics 

function get_mean_std_dev(x::AbstractArray; dims = ndims(x))
    μ = mean(x, dims = dims)
    σ = std(x, dims = dims, corrected = false)
    return μ, σ
end

function mean_loglikelihood(preds, labels)
    return mean(log(p[s]) for (s, p) in zip(labels, preds))
end

function brier_score(preds, labels)
    return mean(
        sum(abs2(pi - (i == s)) for (i, pi) in enumerate(p)) for
        (s, p) in zip(labels, preds)
    )
end

function ExpectedCalibrationError(preds, labels, num_bins = 10;)
    preds = [x for x in eachcol(preds)]
    ece_estimator = ECE(UniformBinning(num_bins), (μ, y) -> kl_divergence(y, μ))
    ece = ece_estimator(preds, labels)
    return ece
end


function prediction_metrics(preds, labels)
    metrics = Dict()
    preds = [x for x in eachcol(preds)]
    # Calulate mean loglikelihood 
    metrics["mean_log_likelihood"] = mean_loglikelihood(preds, labels)
    # Brier Score 
    metrics["brier_score"] = brier_score(preds, labels)
    return metrics
end
