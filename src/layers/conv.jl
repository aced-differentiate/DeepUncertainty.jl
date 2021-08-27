using Flux
using Random
using Flux: @functor, glorot_uniform, glorot_normal, create_bias, batch

include("../initializers.jl")

"""
    MCConv(filter, in => out, σ = identity;
            stride = 1, pad = 0, dilation = 1, groups = 1, [bias, weight, init])
    MCConv(layer, dropout_rate)

Creates a traditional Conv layer with MC dropout functionality. 
MC Dropout simply means that dropout is activated in both train and test times 

Reference - Dropout as a bayesian approximation - https://arxiv.org/abs/1506.02142 

The traditional conv layer is a field in the struct MCConv, so all the 
arguments required for the conv layer can be provided, or the layer can 
be provided too. The forward pass is the conv operation of the conv
layer followed by dropout applied on the resulting activations. 

    y = dropout(Conv(x), dropout_rate)

# Fields
- `layer`: A traditional conv layer 
- `dropout_rate::AbstractFloat`: Dropout rate 

# Arguments 
- `filter::NTuple{N,Integer}`: Kernel dimensions, eg, (5, 5) 
- `ch::Pair{<:Integer,<:Integer}`: Input channels => output channels 
- `dropout_rate::AbstractFloat`: Dropout rate 
- `σ::F=identity`: Activation function, defaults to identity
- `init=glorot_normal`: Initialization function, defaults to glorot_normal 
"""
struct MCConv{L,F}
    layer::L
    dropout_rate::F
    function MCConv(layer::L, dropout_rate::F) where {L,F}
        new{typeof(layer),F}(layer, dropout_rate)
    end
end

function MCConv(
    k::NTuple{N,Integer},
    ch::Pair{<:Integer,<:Integer},
    dropout_rate::F,
    σ = identity;
    init = glorot_uniform,
    stride = 1,
    pad = 0,
    dilation = 1,
    groups = 1,
    bias = true,
) where {N,F}

    layer = Flux.Conv(
        k,
        ch,
        σ;
        init = init,
        stride = stride,
        pad = pad,
        dilation = dilation,
        groups = groups,
        bias = bias,
    )
    return MCConv(layer, dropout_rate)
end

@functor MCConv

function (c::MCConv)(x::AbstractArray; dropout::Bool = true)
    # Conv Batch Ensemble params 
    output = c.layer(x)
    output = Flux.dropout(output, c.dropout_rate; active = dropout)
    return output
end

"""
    ConvBatchEnsemble(filter, in => out, rank, 
                    ensemble_size, σ = identity;
                    stride = 1, pad = 0, dilation = 1, 
                    groups = 1, [bias, weight, init])
    ConvBatchEnsemble(layer, alpha, gamma, ensemble_bias, ensemble_act, rank)

Creates a conv BatchEnsemble layer. Batch ensemble is a memory efficient alternative 
for deep ensembles. In deep ensembles, if the ensemble size is N, N different models 
are trained, making the time and memory complexity O(N * complexity of one network). 
BatchEnsemble generates weight matrices for each member in the ensemble using a 
couple of rank 1 vectors R (alpha), S (gamma), RS' and multiplying the result with 
weight matrix W element wise. We also call R and S as fast weights. 

Reference - https://arxiv.org/abs/2002.06715 

During both training and testing, we repeat the samples along the batch dimension 
N times, where N is the ensemble_size. For example, if each mini batch has 10 samples 
and our ensemble size is 4, then the actual input to the layer has 40 samples. 
The output of the layer has 40 samples as well, and each 10 samples can be considered 
as the output of an esnemble member. 

# Fields 
- `layer`: The dense layer which transforms the pertubed input to output 
- `alpha`: The first Fast weight of size (in_dim, ensemble_size)
- `gamma`: The second Fast weight of size (out_dim, ensemble_size)
- `ensemble_bias`: Bias added to the ensemble output, separate from dense layer bias 
- `ensemble_act`: The activation function to be applied on ensemble output 
- `rank`: Rank of the fast weights (rank > 1 doesn't work on GPU for now)

# Arguments 
- `filter::NTuple{N,Integer}`: Kernel dimensions, eg, (5, 5) 
- `ch::Pair{<:Integer,<:Integer}`: Input channels => output channels 
- `rank::Integer`: Rank of the fast weights 
- `ensemble_size::Integer`: Number of models in the ensemble 
- `σ::F=identity`: Activation of the dense layer, defaults to identity
- `init=glorot_normal`: Initialization function, defaults to glorot_normal 
- `alpha_init=glorot_normal`: Initialization function for the alpha fast weight,
                            defaults to glorot_normal 
- `gamma_init=glorot_normal`: Initialization function for the gamma fast weight, 
                            defaults to glorot_normal 
- `bias::Bool=true`: Toggle the usage of bias in the dense layer 
- `ensemble_bias::Bool=true`: Toggle the usage of ensemble bias 
- `ensemble_act::F=identity`: Activation function for enseble outputs 
"""
struct ConvBatchEnsemble{L,F,M,B}
    layer::L
    alpha::M
    gamma::M
    ensemble_bias::B
    ensemble_act::F
    rank::Integer
    function ConvBatchEnsemble(
        layer::L,
        alpha::M,
        gamma::M,
        ensemble_bias = true,
        ensemble_act::F = identity,
        rank::Integer = 1,
    ) where {M,F,L}
        ensemble_bias = create_bias(gamma, ensemble_bias, size(gamma)[1], size(gamma)[2])
        new{typeof(layer),F,M,typeof(ensemble_bias)}(
            layer,
            alpha,
            gamma,
            ensemble_bias,
            ensemble_act,
            rank,
        )
    end
end

function ConvBatchEnsemble(
    k::NTuple{N,Integer},
    ch::Pair{<:Integer,<:Integer},
    rank::Integer,
    ensemble_size::Integer,
    σ = identity;
    init = glorot_normal,
    alpha_init = glorot_normal,
    gamma_init = glorot_normal,
    stride = 1,
    pad = 0,
    dilation = 1,
    groups = 1,
    bias = true,
    ensemble_bias = true,
    ensemble_act = identity,
) where {N}

    layer = Flux.Conv(
        k,
        ch,
        σ;
        stride = stride,
        pad = pad,
        dilation = dilation,
        init = init,
        groups = groups,
        bias = bias,
    )
    in_dim = ch[1]
    out_dim = ch[2]
    if rank >= 1
        alpha_shape = (in_dim, ensemble_size)
        gamma_shape = (out_dim, ensemble_size)
    else
        error("Rank must be >= 1.")
    end
    alpha = alpha_init(alpha_shape)
    gamma = gamma_init(gamma_shape)

    return ConvBatchEnsemble(layer, alpha, gamma, ensemble_bias, ensemble_act, rank)
end

@functor ConvBatchEnsemble

function (c::ConvBatchEnsemble)(x::AbstractArray)
    # Conv Batch Ensemble params 
    layer = c.layer
    alpha = c.alpha
    gamma = c.gamma
    e_b = c.ensemble_bias
    e_σ = c.ensemble_act

    batch_size = size(x)[end]
    in_size = size(alpha)[1]
    out_size = size(gamma)[1]
    ensemble_size = size(alpha)[2]
    samples_per_model = batch_size ÷ ensemble_size

    # Alpha, gamma shapes - [units, ensembles, rank]
    e_b = repeat(e_b, samples_per_model)
    alpha = repeat(alpha, samples_per_model)
    gamma = repeat(gamma, samples_per_model)
    # Reshape alpha, gamma to [units, batch_size, rank]
    e_b = reshape(e_b, (1, 1, out_size, batch_size))
    alpha = reshape(alpha, (1, 1, in_size, batch_size))
    gamma = reshape(gamma, (1, 1, out_size, batch_size))

    perturbed_x = x .* alpha
    output = layer(perturbed_x) .* gamma
    output = e_σ.(output .+ e_b)

    return output
end

"""
    ConvBayesianBatchEnsemble(filter, in => out, rank, 
                                ensemble_size, σ = identity;
                                stride = 1, pad = 0, dilation = 1, 
                                groups = 1, [bias, weight, init])
    ConvBayesianBatchEnsemble(layer, alpha_sampler, gamma_sampler,
                                ensemble_bias, ensemble_act, rank)

Creates a bayesian conv BatchEnsemble layer. 
Batch ensemble is a memory efficient alternative for deep ensembles. In deep ensembles, 
if the ensemble size is N, N different models are trained, making the time and memory 
complexity O(N * complexity of one network). 
BatchEnsemble generates weight matrices for each member in the ensemble using a 
couple of rank 1 vectors R (alpha), S (gamma), RS' and multiplying the result with 
weight matrix W element wise. We also call R and S as fast weights. In the bayesian 
version of batch ensemble, instead of having point estimates of the fast weights, we 
sample them form a trainable parameterized distribution. 

Reference - https://arxiv.org/abs/2005.07186

During both training and testing, we repeat the samples along the batch dimension 
N times, where N is the ensemble_size. For example, if each mini batch has 10 samples 
and our ensemble size is 4, then the actual input to the layer has 40 samples. 
The output of the layer has 40 samples as well, and each 10 samples can be considered 
as the output of an esnemble member. 

# Fields 
- `layer`: The conv layer which transforms the pertubed input to output 
- `alpha_sampler`: Sampler for the first Fast weight of size (in_dim, ensemble_size)
- `gamma_sampler`: Sampler for the second Fast weight of size (out_dim, ensemble_size)
- `ensemble_bias`: Bias added to the ensemble output, separate from conv layer bias 
- `ensemble_act`: The activation function to be applied on ensemble output 
- `rank`: Rank of the fast weights (rank > 1 doesn't work on GPU for now)

# Arguments 
- `filter::NTuple{N,Integer}`: Kernel dimensions, eg, (5, 5) 
- `ch::Pair{<:Integer,<:Integer}`: Input channels => output channels 
- `rank::Integer`: Rank of the fast weights 
- `ensemble_size::Integer`: Number of models in the ensemble 
- `σ::F=identity`: Activation of the dense layer, defaults to identity
- `init=glorot_normal`: Initialization function, defaults to glorot_normal 
- `alpha_init=glorot_normal`: Initialization function for the alpha fast weight,
                            defaults to TrainableGlorotNormal 
- `gamma_init=glorot_normal`: Initialization function for the gamma fast weight, 
                            defaults to TrainableGlorotNormal 
- `bias::Bool=true`: Toggle the usage of bias in the dense layer 
- `ensemble_bias::Bool=true`: Toggle the usage of ensemble bias 
- `ensemble_act::F=identity`: Activation function for enseble outputs 
"""
struct ConvBayesianBatchEnsemble{L,I,F,B}
    layer::L
    alpha_sampler::I
    gamma_sampler::I
    ensemble_bias::B
    ensemble_act::F
    rank::Integer
    function ConvBayesianBatchEnsemble(
        layer::L,
        alpha_sampler::I,
        gamma_sampler::I,
        ensemble_bias = true,
        ensemble_act::F = identity,
        rank::Integer = 1,
    ) where {I,F,L}
        gamma_shape = size(gamma_sampler.mean)
        ensemble_bias =
            create_bias(gamma_sampler.mean, ensemble_bias, gamma_shape[1], gamma_shape[2])
        new{typeof(layer),typeof(alpha_sampler),F,typeof(ensemble_bias)}(
            layer,
            alpha_sampler,
            gamma_sampler,
            ensemble_bias,
            ensemble_act,
            rank,
        )
    end
end

function ConvBayesianBatchEnsemble(
    k::NTuple{N,Integer},
    ch::Pair{<:Integer,<:Integer},
    rank::Integer,
    ensemble_size::Integer,
    σ = identity;
    init = glorot_uniform,
    alpha_init = TrainableGlorotNormal,
    gamma_init = TrainableGlorotNormal,
    stride = 1,
    pad = 0,
    dilation = 1,
    groups = 1,
    bias = true,
) where {N}

    layer = Flux.Conv(
        k,
        ch,
        σ;
        stride = stride,
        pad = pad,
        dilation = dilation,
        init = init,
        groups = groups,
        bias = bias,
    )
    in_dim = ch[1]
    out_dim = ch[2]
    if rank >= 1
        alpha_shape = (in_dim, ensemble_size)
        gamma_shape = (out_dim, ensemble_size)
    else
        error("Rank must be >= 1.")
    end
    # Initialize alpha and gamma samplers 
    alpha_sampler = TrainableGlorotNormal(
        alpha_shape,
        mean_initializer = alpha_init,
        stddev_initializer = alpha_init,
    )
    gamma_sampler = TrainableGlorotNormal(
        gamma_shape,
        mean_initializer = gamma_init,
        stddev_initializer = gamma_init,
    )

    return ConvBayesianBatchEnsemble(layer, alpha_sampler, gamma_sampler, bias, σ, rank)
end

@functor ConvBayesianBatchEnsemble

function (c::ConvBayesianBatchEnsemble)(x::AbstractArray)
    # Conv Batch Ensemble params 
    layer = c.layer
    # Sample alpha, gamma from a trainable distribution
    alpha = c.alpha_sampler()
    gamma = c.gamma_sampler()
    e_b = c.ensemble_bias
    e_σ = c.ensemble_act

    batch_size = size(x)[end]
    in_size = size(alpha)[1]
    out_size = size(gamma)[1]
    ensemble_size = size(alpha)[2]
    samples_per_model = batch_size ÷ ensemble_size

    # Alpha, gamma shapes - [units, ensembles, rank]
    e_b = repeat(e_b, samples_per_model)
    alpha = repeat(alpha, samples_per_model)
    gamma = repeat(gamma, samples_per_model)
    # Reshape alpha, gamma to [units, batch_size, rank]
    e_b = reshape(e_b, (1, 1, out_size, batch_size))
    alpha = reshape(alpha, (1, 1, in_size, batch_size))
    gamma = reshape(gamma, (1, 1, out_size, batch_size))

    perturbed_x = x .* alpha
    output = layer(perturbed_x) .* gamma
    output = e_σ.(output .+ e_b)

    return output
end
