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

function calculate_entropy(probs, eps = 1e-6; from_logits = true)
    if from_logits
        probs = softmax(probs, dims = 1)
    end
    return -sum(probs .* log.(probs .+ eps), dims = 1)
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

function expected_calibration_error(preds, labels, num_bins = 10; from_logits = true)
    if from_logits
        preds = softmax(preds, dims = 1)
    end
    preds = cpu(preds)
    labels = cpu(labels)
    preds = [x for x in eachcol(preds)]
    ece_estimator = ECE(UniformBinning(num_bins), SqEuclidean())
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
