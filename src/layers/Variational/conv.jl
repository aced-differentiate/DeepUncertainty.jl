"""
    ConvBatchEnsemble(filter, in => out, σ = identity;
                    stride = 1, pad = 0, dilation = 1, 
                    groups = 1, [bias, weight, init])

Creates a variational conv layer. Computes variational bayesian 
approximation to the distribution over the parameters of the conv layer. 
The stochasticity is during the forward pass, instead of using point 
estimates for weights and biases, we sample from the distribution over 
weights and biases. Gradients of the distribution's learnable parameters 
are trained using the reparameterization trick.

Reference - https://arxiv.org/abs/1505.05424 
We use DistributionsAD - https://github.com/TuringLang/DistributionsAD.jl 
to help us with backprop. 

# Fields 
- `σ`: Activation function, applies to logits after layer transformation 
- `weight_sampler`: A trainable distribution from which weights are sampled 
                    in every forward pass 
- `bias_sampler`: A trainable distribution from which biases are sampled in 
                    every forward pass 
- `stride`: Convolution stride 
- `pad`
- `dilation`
- `groups`

# Arguments 
- `filter::NTuple{N,Integer}`: Kernel dimensions, eg, (5, 5) 
- `ch::Pair{<:Integer,<:Integer}`: Input channels => output channels 
- `σ::F=identity`: Activation of the dense layer, defaults to identity
- `weight_dist=TrainableMvNormal`: Initialization function for weights.  
- `bias_dist=TrainableMvNormal`: Initialization function for biases. 

"""
struct VariationalConv{N,M,F,A,V}
    σ::F
    weight_sampler::A
    bias_sampler::V
    stride::NTuple{N,Int}
    pad::NTuple{M,Int}
    dilation::NTuple{N,Int}
    groups::Int
end

function VariationalConv(
    k::NTuple{N,Integer},
    ch::Pair{<:Integer,<:Integer},
    σ = identity;
    stride = 1,
    pad = 0,
    dilation = 1,
    groups = 1,
    init = glorot_normal,
    weight_dist = TrainableMvNormal,
    bias_dist = TrainableMvNormal,
    weight = Flux.convfilter(k, (ch[1] ÷ groups => ch[2])),
    bias = true,
) where {N}

    stride = Flux.expand(Val(N), stride)
    dilation = Flux.expand(Val(N), dilation)
    pad = Flux.calc_padding(VariationalConv, pad, size(weight)[1:N], dilation, stride)
    bias = create_bias(weight, bias, size(weight, N + 2))
    # Distribution from which weights are sampled 
    weight_sampler = weight_dist((k..., ch...), init = init)
    # Distribution from which biases are sampled 
    bias_sampler = bias_dist(size(bias), init = init)
    return VariationalConv(σ, weight_sampler, bias_sampler, stride, pad, dilation, groups)
end

@functor VariationalConv (weight_sampler, bias_sampler)

function (c::VariationalConv)(x)
    # Sample weights and biases 
    weight = c.weight_sampler()
    bias = c.bias_sampler()
    σ, b = c.σ, reshape(bias, ntuple(_ -> 1, length(c.stride))..., :, 1)
    output = Flux.NNlib.conv(
        x,
        weight;
        stride = c.stride,
        pad = c.pad,
        dilation = c.dilation,
        groups = c.groups,
    )
    σ.(output .+ b)
end
