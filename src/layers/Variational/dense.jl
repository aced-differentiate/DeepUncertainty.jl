"""
    VariationalDense(in, out, σ=identity;
                    weight_init=TrainableDistribution, 
                    bias_init=TrainableDistribution, 
                    bias=true)
    VariationalDense(weight_sampler, bias_sampler, act)

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
- `init`: Distribution parameters Initialization, defaults to glorot_normal
- `weight_dist`: Weight distribution, defaults to a trainable multivariate normal
- `bias_dist`: Bias distribution, defaults to trainable multivariate normal

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
    init = glorot_normal,
    weight_dist = TrainableMvNormal,
    bias_dist = TrainableMvNormal,
    device = cpu,
)
    # Initialize alpha and gamma samplers 
    weight_sampler = weight_dist((out, in), init = init, device = device)
    bias_sampler = bias_dist((out,), init = init, device = device)

    return VariationalDense(weight_sampler, bias_sampler, σ)
end

@functor VariationalDense

function (dv::VariationalDense)(x)
    weight = dv.weight_sampler()
    bias = dv.bias_sampler()
    logits = weight * x .+ bias
    return dv.act.(logits)
end
