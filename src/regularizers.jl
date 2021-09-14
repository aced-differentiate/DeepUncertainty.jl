function NormalKLDivergence(layer)
    μ1 = layer.mean_constraint.(layer.mean)
    σ1 = layer.stddev_constraint.(layer.stddev)

    # Calculate prior loglikelihood 
    μ2 = gpu(zeros(prod(layer.shape)))
    σ2 = gpu(ones(prod(layer.shape)))

    # compute the KL 
    kl = log.(σ2 ./ σ1)
    kl += (σ1 .^ 2 .+ (μ1 .- μ2) .^ 2) ./ 2 .* (σ2) .^ 2
    kl = sum(kl) - 0.5
    return layer.complexity_weight * kl
end
