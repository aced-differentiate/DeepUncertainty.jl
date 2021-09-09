function KLDivergence(layer::L) where {L}
    # dist = DistributionsAD.TuringMvNormal(layer.mean, layer.stddev)
    mean = layer.mean_constraint.(layer.mean)
    stddev = layer.stddev_constraint.(layer.stddev)
    sample = gpu(layer.sample)
    posterior = DistributionsAD.TuringMvNormal(mean, stddev)
    posterior_loglikelihood = DistributionsAD.loglikelihood(posterior, sample)

    # Calculate prior loglikelihood 
    mean = gpu(zeros(prod(layer.shape)))
    stddev = gpu(ones(prod(layer.shape))) 
    prior = DistributionsAD.TuringMvNormal(mean, stddev)
    prior_loglikelihood = DistributionsAD.loglikelihood(prior, sample)

    return layer.complexity_weight * (posterior_loglikelihood - prior_loglikelihood)
end 