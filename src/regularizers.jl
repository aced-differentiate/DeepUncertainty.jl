normal_kl_divergence(layer::Any) = 0
scale_mixture_kl_divergence(layer::Any) = 0

function normal_kl_divergence(layer::AbstractTrainableDist)
    posterior = DistributionsAD.TuringMvNormal(layer.mean, abs.(layer.stddev))
    posterior_loglikelihood = DistributionsAD.loglikelihood(posterior, layer.sample)

    # Calculate prior loglikelihood 
    μ2 = gpu(zeros(prod(layer.shape)))
    σ2 = gpu(ones(prod(layer.shape)))
    prior = DistributionsAD.TuringMvNormal(μ2, σ2)
    prior_loglikelihood = DistributionsAD.loglikelihood(prior, layer.sample)

    return (posterior_loglikelihood - prior_loglikelihood)
end

function scale_mixture_kl_divergence(
    layer::AbstractTrainableDist;
    pi = 0.5,
    prior_sigma1 = 1.0,
    prior_sigma2 = 0.0025,
)
    posterior = DistributionsAD.TuringMvNormal(layer.mean, abs.(layer.stddev))
    posterior_loglikelihood = DistributionsAD.loglikelihood(posterior, layer.sample)

    μ = gpu(zeros(prod(layer.shape)))
    σ1 = gpu(ones(prod(layer.shape)) .* prior_sigma1)
    σ2 = gpu(ones(prod(layer.shape)) .* prior_sigma2)
    prior = DistributionsAD.TuringMvNormal(μ, σ1)
    likelihood_n1 = exp.(DistributionsAD.loglikelihood(prior, layer.sample))
    prior = DistributionsAD.TuringMvNormal(μ, σ2)
    likelihood_n2 = exp.(DistributionsAD.loglikelihood(prior, layer.sample))
    p_scalemixture = pi * likelihood_n1 + (1 - pi) * likelihood_n2
    prior_loglikelihood = log(p_scalemixture)

    return (posterior_loglikelihood - prior_loglikelihood)
end
