"""
    VariationalDense(in, out, σ=identity;
                    weight_init=TrainableDistribution, 
                    bias_init=TrainableDistribution, 
                    bias=true)

Creates a variational dense layer. Computes variational bayesian 
approximation to the distribution over the parameters of the dense layer. 
The stochasticity is during the forward pass, instead of using point 
estimates for weights and biases, we sample from the distribution over 
weights and biases. Gradients of the distribution's learnable parameters 
are trained using the reparameterization trick.

Reference - https://arxiv.org/abs/1505.05424 
We use DistributionsAD - https://github.com/TuringLang/DistributionsAD.jl 
to help us with backprop. 

# Fields 
- `weight_sampler`: A trainable distribution from which weights are sampled 
                    in every forward pass 
- `bias_sampler`: A trainable distribution from which biases are sampled in 
                    every forward pass 
- `act`: Activation function, applies to logits after layer transformation 

# Arguents 
- `in::Integer`: Input dimension size 
- `out::Integer`: Output dimension size 
- `σ`: Acivation function, defaults to identity 
- `weight_init`: Weight initializer, defaults to a trainable multivariate normal
- `bias_init`: Bias initializer, defaults to trainable multivariate normal 
- `bias`: Use bias or not 
"""
struct VariationalDense{WS,BS,F}
    weight_sampler::WS
    bias_sampler::BS
    act::F
end

function VariationalDense(
    in::Integer,
    out::Integer,
    σ = identity;
    weight_init = TrainableDistribution,
    bias_init = TrainableDistribution,
    complexity_weight = 1e-5,
    mean_init = glorot_normal,
    stddev_init = glorot_normal,
    mean_constraint = identity,
    stddev_constraint = softplus,
    prior_distribution = DistributionsAD.TuringMvNormal,
    posterior_distribution = DistributionsAD.TuringMvNormal,
    bias = true,
)
    # Initialize alpha and gamma samplers 
    weight_sampler = weight_init(
        (out, in),
        complexity_weight = complexity_weight,
        mean_init = mean_init,
        stddev_init = stddev_init,
        mean_constraint = mean_constraint,
        stddev_constraint = stddev_constraint,
        prior_distribution = prior_distribution,
        posterior_distribution = posterior_distribution,
    )
    bias_sampler = bias_init(
        (out,),
        complexity_weight = complexity_weight,
        mean_init = mean_init,
        stddev_init = stddev_init,
        mean_constraint = mean_constraint,
        stddev_constraint = stddev_constraint,
        prior_distribution = prior_distribution,
        posterior_distribution = posterior_distribution,
    )

    return VariationalDense(weight_sampler, bias_sampler, σ)
end

@functor VariationalDense

function (dv::VariationalDense)(x)
    weight = dv.weight_sampler()
    bias = dv.bias_sampler()
    logits = weight * x .+ bias
    return dv.act.(logits)
end
