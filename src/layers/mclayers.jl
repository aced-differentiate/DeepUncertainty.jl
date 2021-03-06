using Flux
using Random
using Test
using Flux: @functor

"""
    MCLayer(layer, dropout)
A generic Monte Carlo dropout layer. Takes in any "traditional" flux 
layer and a function that implements dropout. Performs the usual layer 
forward pass and then passes the acitvations through the given dropout function. 
Let x be the input array, the forward pass then becomes -- dropout(layer(x))
    
# Fields
- `layer`: Traditional Flux layer 
- `dropout`: A function that implements dropout 
"""
struct MCLayer{L,F}
    layer::L
    dropout::F
end

@functor MCLayer

"""
    MCDense(in::Int, out::Int, dropout_rate, σ=identity; bias=true, init=glorot_uniform)
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
- `dropout`: A function that implements dropout  

# Arguments 
- `in::Integer`: Input dimension of features 
- `out::Integer`: Output dimension of features 
- `dropout_rate::AbstractFloat`: Dropout rate 
- `σ::F=identity`: Activation function, defaults to identity
- `init=glorot_normal`: Initialization function, defaults to glorot_normal 
"""
function MCDense(in::Integer, out::Integer, dropout_rate, σ = identity, kwargs...)
    layer = Flux.Dense(in, out, σ; kwargs...)
    dropout = (x; k...) -> Flux.dropout(x, dropout_rate; k...)
    return MCLayer(layer, dropout)
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
function MCConv(
    k::NTuple{N,Integer},
    ch::Pair{<:Integer,<:Integer},
    dropout_rate,
    σ = identity;
    kwargs...,
) where {N}
    layer = Flux.Conv(k, ch, σ; kwargs...)
    dropout = (x; k...) -> Flux.dropout(x, dropout_rate; k...)
    return MCLayer(layer, dropout)
end

function MCConv(
    w::AbstractArray{T,N},
    bias,
    dropout_rate,
    σ = identity,
    kwargs...,
) where {T,N}
    layer = Flux.Conv(w, bias, σ, kwargs...)
    dropout = (x; k...) -> Flux.dropout(x, dropout_rate; k...)
    return MCLayer(layer, dropout)
end

"""
The forward pass of a MC layer: Passes the input through the 
usual layer first and then through a dropout layer. 

# Arguments 
- `x`: Input tensors 
- `dropout=true`: Toggle to control dropout, it's preferred to keep 
dropout always on, but just in case if it's needed. 
"""
function (mc::MCLayer)(x; dropout = true)
    # Layer forward pass 
    # Dropout on activations 
    output = mc.dropout(mc.layer(x); active = dropout)
    return output
end
