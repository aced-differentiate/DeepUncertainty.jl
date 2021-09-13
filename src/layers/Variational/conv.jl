# pad dims of x with dims of y until ndims(x) == ndims(y)
_paddims(x::Tuple, y::Tuple) = (x..., y[(end-(length(y)-length(x)-1)):end]...)

expand(N, i::Tuple) = i
expand(N, i::Integer) = ntuple(_ -> i, N)


struct SamePad end

calc_padding(lt, pad, k::NTuple{N,T}, dilation, stride) where {T,N} =
    expand(Val(2 * N), pad)
function calc_padding(lt, ::SamePad, k::NTuple{N,T}, dilation, stride) where {N,T}
    # Ref: "A guide to convolution arithmetic for deep learning" https://arxiv.org/abs/1603.07285

    # Effective kernel size, including dilation
    k_eff = @. k + (k - 1) * (dilation - 1)
    # How much total padding needs to be applied?
    pad_amt = @. k_eff - 1
    # In case amount of padding is odd we need to apply different amounts to each side.
    return Tuple(mapfoldl(i -> [cld(i, 2), fld(i, 2)], vcat, pad_amt))
end

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

# Arguments 
- `filter::NTuple{N,Integer}`: Kernel dimensions, eg, (5, 5) 
- `ch::Pair{<:Integer,<:Integer}`: Input channels => output channels 
- `σ::F=identity`: Activation of the dense layer, defaults to identity
- `weight_init=TrainableDistribution`: Initialization function for weights.  
- `bias_init=TrainableDistribution`: Initialization function for biases. 
- `complexity_weight=1e-5`: The reguralization constant to be multiplied with 
                            KL Divergence between prior and posterior distributions. 
- `mean_init=glorot_normal`: Initialization function for mean tensor. 
- `stddev_init=glorot_normal`: Initialization function for stddev tensor. 
- `mean_constraint=identity`: Constraint to be applied on mean tensor. 
- `stddev_constraint=softplus`: Constraint to be applied on mean tensor. 
- `prior_distribution=TuringMvNormal`: Prior distribution on the parameters. 
- `posterior_distribution=TuringMvNormal`: Poterior distribution on the parameters.
- `bias::Bool=true`: Toggle the usage of bias in the dense layer 

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
    weight_init = TrainableDistribution,
    bias_init = TrainableDistribution,
    complexity_weight = 1e-5,
    mean_init = glorot_normal,
    stddev_init = glorot_normal,
    mean_constraint = identity,
    stddev_constraint = softplus,
    prior_distribution = DistributionsAD.TuringMvNormal,
    posterior_distribution = DistributionsAD.TuringMvNormal,
    weight = convfilter(k, (ch[1] ÷ groups => ch[2]); init = glorot_normal),
    bias = true,
) where {N}

    stride = expand(Val(N), stride)
    dilation = expand(Val(N), dilation)
    pad = calc_padding(VariationalConv, pad, size(weight)[1:N], dilation, stride)
    bias = create_bias(weight, bias, size(weight, N + 2))
    # Distribution from which weights are sampled 
    weight_sampler = weight_init(
        (k..., ch...),
        complexity_weight = complexity_weight,
        mean_init = mean_init,
        stddev_init = stddev_init,
        mean_constraint = mean_constraint,
        stddev_constraint = stddev_constraint,
        prior_distribution = prior_distribution,
        posterior_distribution = posterior_distribution,
    )
    # Distribution from which biases are sampled 
    bias_sampler = bias_init(
        size(bias),
        complexity_weight = complexity_weight,
        mean_init = mean_init,
        stddev_init = stddev_init,
        mean_constraint = mean_constraint,
        stddev_constraint = stddev_constraint,
        prior_distribution = prior_distribution,
        posterior_distribution = posterior_distribution,
    )
    return VariationalConv(σ, weight_sampler, bias_sampler, stride, pad, dilation, groups)
end

convfilter(
    filter::NTuple{N,Integer},
    ch::Pair{<:Integer,<:Integer};
    init = glorot_uniform,
) where {N} = init(filter..., ch...)

@functor VariationalConv

function (c::VariationalConv)(x)
    # Sample weights and biases 
    weight = c.weight_sampler()
    bias = c.bias_sampler()
    σ, b = c.σ, reshape(bias, ntuple(_ -> 1, length(c.stride))..., :, 1)
    cdims = DenseConvDims(
        x,
        weight;
        stride = c.stride,
        padding = c.pad,
        dilation = c.dilation,
        groups = c.groups,
    )
    σ.(conv(x, weight, cdims) .+ b)
end
