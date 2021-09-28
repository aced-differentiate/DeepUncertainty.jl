using Flux
using Flux: glorot_uniform, normalise, @functor# , destructure
using Zygote: @adjoint, @nograd
using LinearAlgebra, SparseArrays
using Statistics
using ChemistryFeaturization
using DeepUncertainty

struct MCAGNConv{W,F}
    selfweight::W
    convweight::W
    bias::W
    σ::F
    dropout
end

function MCAGNConv(
    ch::Pair{<:Integer,<:Integer},
    dropout_rate = 0.1,
    σ = softplus;
    initW = glorot_uniform,
    initb = zeros,
    T::DataType = Float64,
)
    selfweight = T.(initW(ch[2], ch[1]))
    convweight = T.(initW(ch[2], ch[1]))
    b = T.(initb(ch[2], 1))
    dropout = (x; k...) -> Flux.dropout(x, dropout_rate; k...)
    MCAGNConv(selfweight, convweight, b, σ, dropout)
end

@functor MCAGNConv

"""
 Define action of layer on inputs: do a graph convolution, add this (weighted by convolutional weight) to the features themselves (weighted by self weight) and the per-feature bias (concatenated to match number of nodes in graph).
# Arguments
- input: a FeaturizedAtoms object, or graph_laplacian, encoded_features
# Note
In the case of providing two matrices, the following conditions must hold:
- `lapl` must be square and of dimension N x N where N is the number of nodes in the graph
- `X` (encoded features) must be of dimension M x N, where M is `size(l.convweight)[2]` (or equivalently, `size(l.selfweight)[2]`)
"""
function (l::MCAGNConv)(lapl, X)
    # should we put dimension checks here? Could allow more informative errors, but would likely introduce performance penalty. For now it's just in docstring.
    out_mat = normalise(
        l.σ.(
            l.convweight * X * lapl +
            l.selfweight * X +
            reduce(hcat, l.bias for i = 1:size(X, 2)),
        ),
        dims = [1, 2],
    )
    out_mat = l.dropout(out_mat)
    lapl, out_mat
end

# alternate signature so FeaturizedAtoms can be fed into first layer
(l::MCAGNConv)(a::FeaturizedAtoms{AtomGraph,GraphNodeFeaturization}) =
    l(a.atoms.laplacian, a.encoded_features)

# signature to splat appropriately
(l::MCAGNConv)(t::Tuple{Matrix{R1},Matrix{R2}}) where {R1<:Real,R2<:Real} = l(t...)

# Bayesian Atomic Graph CNN 
struct VariationalAGNConv{W,F}
    selfweight_sampler::W
    convweight_sampler::W
    bias_smapler::W
    σ::F
end

function VariationalAGNConv(
    ch::Pair{<:Integer,<:Integer},
    σ = softplus;
    initW = glorot_uniform,
    initb = zeros,
    T::DataType = Float64,
)
    selfweight = T.(initW(ch[2], ch[1]))
    selfweight_sampler = TrainableMvNormal(size(selfweight))
    convweight = T.(initW(ch[2], ch[1]))
    convweight_sampler = TrainableMvNormal(size(convweight))
    b = T.(initb(ch[2], 1))
    bias_sampler = TrainableMvNormal(size(b))
    VariationalAGNConv(selfweight_sampler, convweight_sampler, bias_sampler, σ)
end

@functor VariationalAGNConv

"""
 Define action of layer on inputs: do a graph convolution, add this (weighted by convolutional weight) to the features themselves (weighted by self weight) and the per-feature bias (concatenated to match number of nodes in graph).
# Arguments
- input: a FeaturizedAtoms object, or graph_laplacian, encoded_features
# Note
In the case of providing two matrices, the following conditions must hold:
- `lapl` must be square and of dimension N x N where N is the number of nodes in the graph
- `X` (encoded features) must be of dimension M x N, where M is `size(l.convweight)[2]` (or equivalently, `size(l.selfweight)[2]`)
"""
function (l::VariationalAGNConv)(lapl, X)
    # should we put dimension checks here? Could allow more informative errors, but would likely introduce performance penalty. For now it's just in docstring.
    convweight = l.convweight_sampler()
    selfweight = l.selfweight_sampler()
    bias = l.bias_smapler()
    out_mat = normalise(
        l.σ.(
            convweight * X * lapl +
            selfweight * X +
            reduce(hcat, bias for i = 1:size(X, 2)),
        ),
        dims = [1, 2],
    )
    lapl, out_mat
end

# alternate signature so FeaturizedAtoms can be fed into first layer
(l::VariationalAGNConv)(a::FeaturizedAtoms{AtomGraph,GraphNodeFeaturization}) =
    l(a.atoms.laplacian, a.encoded_features)

# signature to splat appropriately
(l::VariationalAGNConv)(t::Tuple{Matrix{R1},Matrix{R2}}) where {R1<:Real,R2<:Real} = l(t...)
