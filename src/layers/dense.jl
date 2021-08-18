using Flux 
using Flux:@functor, glorot_uniform, create_bias

struct DenseBatchEnsemble{L,F,M <: AbstractMatrix,B}
    layer::L 
    α::M
    γ::M
    ensemble_bias::B
    ensemble_σ::F
    rank::Integer
    function DenseBatchEnsemble(layer::L, α::M, γ::M, 
                                ensemble_bias=true, 
                                ensemble_σ::F=identity, 
                                rank::Integer=1) where {M <: AbstractMatrix,F,L}
        ensemble_bias = create_bias(γ, ensemble_bias, size(γ)[1], size(γ)[2])
        new{typeof(layer),F,M,typeof(ensemble_bias)}(layer, α, γ, 
                                                    ensemble_bias, 
                                                    ensemble_σ, 
                                                    rank)
    end
end
  
function DenseBatchEnsemble(in::Integer, out::Integer, 
                            rank::Integer, ensemble_size::Integer, 
                            σ=identity;
                            init=glorot_uniform, 
                            bias=true)
    layer = Flux.Dense(in, out; init=init, bias=bias)
    if rank > 1 
        α_shape = (in, ensemble_size, rank)
        γ_shape = (out, ensemble_size, rank)
    elseif rank == 1 
        α_shape = (in, ensemble_size)
        γ_shape = (out, ensemble_size) 
    else
        error("Rank must be >= 1.")
    end 
    α = init(α_shape) 
    γ = init(γ_shape)
  
    return DenseBatchEnsemble(layer, α, γ, bias, σ, rank)
end
  
@functor DenseBatchEnsemble
  
function (a::DenseBatchEnsemble)(x::AbstractVecOrMat)
    layer = a.layer 
    α = a.α
    γ = a.γ
    e_b = a.ensemble_bias 
    e_σ = a.ensemble_σ
    rank = a.rank

    batch_size = size(x)[end]
    in_size = size(α)[1]
    out_size = size(γ)[1]
    ensemble_size = size(α)[2]
    samples_per_model = batch_size ÷ ensemble_size 

    # TODO: Implement Rank > 1 computations 
    # Reshape the inputs, α and γ
    x = reshape(x, (in_size, samples_per_model, ensemble_size))
    α = reshape(α, (in_size, 1, ensemble_size))
    γ = reshape(γ, (out_size, 1, ensemble_size))
    e_b = reshape(e_b, (out_size, 1, ensemble_size))
    # Perturb the inputs 
    perturbed_x = x .* α;
    # Dense layer forward pass 
    outputs = layer(perturbed_x)
    outputs = outputs .* γ
    outputs = e_σ(outputs .+ e_b)
    outputs = reshape(outputs, (out_size, batch_size))
    return outputs
end
  
(a::DenseBatchEnsemble)(x::AbstractArray) = 
    reshape(a(reshape(x, size(x, 1), :)), :, size(x)[2:end]...)
  
function Base.show(io::IO, l::DenseBatchEnsemble)
    print(io, "DenseBatchEnsemble(", size(l.weight, 2), ", ", size(l.weight, 1))
    l.σ == identity || print(io, ", ", l.σ)
    l.bias == Zeros() && print(io, "; bias=false")
    print(io, ")")
end