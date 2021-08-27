using Flux
using Random
using Flux: @functor, glorot_normal

struct MCdropout{L,F}
    layer::L
    dropout_rate::F
    function MCdropout(layer::L, dropout_rate::F) where {L,F}
        new{typeof(layer),typeof(dropout_rate)}(layer, dropout_rate)
    end
end

function MCdropout(layer::L, dropout_rate::AbstractFloat) where {L}
    return MCdropout(layer, dropout_rate)
end

@functor MCdropout

function (a::MCdropout)(x::AbstractVecOrMat; dropout = true)
    output = a.layer(x)
    output = Flux.dropout(output, a.dropout_rate; active = dropout)
    return output
end
