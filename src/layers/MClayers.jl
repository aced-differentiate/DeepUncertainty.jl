using Flux
using Random
using Flux: @functor, glorot_uniform, create_bias

"""
    MCDense(in, out, dropout_rate, σ=identity; bias=true, init=glorot_uniform)
    MCDense(layer, dropout_rate)

Creates a traditional dense layer with MC dropout functionality. 
MC Dropout simply means that dropout is activated in both train and test times 

Reference - Dropout as a bayesian approximation - https://arxiv.org/abs/1506.02142 

The traditional dense layer is a field in the struct MCDense, so all the 
arguments required for the dense layer can be provided, or the layer can 
be provided too. The forward pass is the affine transformation of the dense
layer followed by dropout applied on the resulting activations. 

    y = dropout(σ.(W * x .+ bias), dropout_rate)

# Fields
- `layer`: A traditional dense layer 
- `dropout_rate::AbstractFloat`: Dropout rate 

# Arguments 
- `in::Integer`: Input dimension of features 
- `out::Integer`: Output dimension of features 
- `dropout_rate::AbstractFloat`: Dropout rate 
- `σ::F=identity`: Activation function, defaults to identity
- `init=glorot_normal`: Initialization function, defaults to glorot_normal 
"""
struct MCDense{L,F}
    layer::L
    dropout_rate::F
    function MCDense(layer::L, dropout_rate::F) where {L,F}
        new{typeof(layer),typeof(dropout_rate)}(layer, dropout_rate)
    end
end

function MCDense(
    in::Integer,
    out::Integer,
    dropout_rate::AbstractFloat,
    σ = identity;
    init = glorot_normal,
    bias = true,
)

    layer = Flux.Dense(in, out, σ; init = init, bias = bias)
    return MCDense(layer, dropout_rate)
end

@functor MCDense

"""
The forward pass of a MCDense layer: Passes the input through the 
usual dense layer first and then through a dropout layer. 

# Arguments 
- `x::AbstractVecOrMat`: Input tensors 
- `dropout::Bool=true`: Toggle to control dropout, it's preferred to keep 
dropout always on, but just in case if it's needed. 
"""
function (a::MCDense)(x::AbstractVecOrMat; dropout = true)
    output = a.layer(x)
    output = Flux.dropout(output, a.dropout_rate; active = dropout)
    return output
end

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
