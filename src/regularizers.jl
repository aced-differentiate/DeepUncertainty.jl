function KLDivergence(layer)
    # dist = DistributionsAD.TuringMvNormal(layer.mean, layer.stddev)
    sample = layer.sample
    mean = layer.mean_constraint.(layer.mean)
    stddev = layer.stddev_constraint.(layer.stddev)
    posterior = layer.posterior_distribution(mean, stddev)
    posterior_loglikelihood = DistributionsAD.loglikelihood(posterior, sample)

    # Calculate prior loglikelihood 
    mean = gpu(zeros(prod(layer.shape)))
    stddev = gpu(ones(prod(layer.shape)))
    prior = layer.prior_distribution(mean, stddev)
    prior_loglikelihood = DistributionsAD.loglikelihood(prior, sample)

    return layer.complexity_weight * (posterior_loglikelihood - prior_loglikelihood)
end
