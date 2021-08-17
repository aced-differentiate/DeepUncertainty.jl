using Flux 
using Flux:@functor, glorot_normal, create_bias

struct DenseBatchEnsemble{F,M <: AbstractMatrix,B}
    weight::M
    bias::B
    σ::F
    function Dense(W::M, bias=true, σ::F=identity) where {M <: AbstractMatrix,F}
        b = create_bias(W, bias, size(W, 1))
        new{F,M,typeof(b)}(W, b, σ)
    end
end

# struct DenseBatchEnsemble{F,M,B}
#     weight::M
#     alpha::M
#     gamma::M 
#     bias::B
#     ensemble_bias::B 
#     act::F 
#     ensemble_act::F 
#     rank::Int 

#     # Constructor 
#     function DenseBatchEnsemble(W::M, alpha::M, γ::M,
#                                 bias=true, 
#                                 ensemble_bias=true, 
#                                 act::F=identity, 
#                                 ensemble_act::F=identity, 
#                                 rank::Int=1)
#         b = create_bias(W, bias, size(W, 1))
#         ensemble_b = create_bias(γ, ensemble_bias, size(γ)[1], size(γ)[2])
#         new{F,M,typeof(b)}(W, alpha, γ, b, ensemble_b, act, ensemble_act, rank)
#     end 
# end 

function DenseBatchEnsemble(in::Integer, 
    out::Integer, 
    rank::Integer, 
    ensemble_size::Integer, 
    init=glorot_normal, 
    α_init=glorot_normal, 
    γ_init=glorot_normal, 
    bias=true, 
    ensemble_bias=true, 
    σ=identity; 
    ensmeble_σ=identity)
    W = init(out, in)
    if rank > 1 
        α_shape = (in, ensemble_size, rank)
        γ_shape = (out, ensemble_size, rank)
    elseif rank == 1 
        α_shape = (in, ensemble_size)
        γ_shape = (out, ensemble_size) 
    else
        error("Rank must be >= 1.")
    end 
    α = α_init(α_shape) 
    γ = γ_init(γ_shape)

    return DenseBatchEnsemble(W, α, γ, 
                            bias, ensemble_bias, 
                            σ, ensmeble_σ, rank)
end 

@functor DenseBatchEnsemble
# Definition of the foward pass 
function (layer::DenseBatchEnsemble)(x::AbstractVecOrMat)
    W = layer.weight 
    α = layer.alpha
    γ = layer.gamma
    b = layer.bias 
    e_b = layer.ensemble_bias 
    σ = layer.act
    e_σ = layer.ensemble_act
    rank = layer.rank
    
    batch_size = size(x)[end]
    in_size = size(α)[1]
    out_size = size(γ)[1]
    ensemble_size = size(α)[2]
    samples_per_model = batch_size ÷ ensemble_size 

    # Reshape the inputs, α and γ
    x = reshape(x, (in_size, samples_per_model, ensemble_size))
    α = reshape(α, (in_size, 1, ensemble_size))
    γ = reshape(γ, (out_size, 1, ensemble_size))

    # Perturb the inputs 
    perturbed_x = x .* α;
    # Dense layer forward pass 
    output = σ(W * perturbed_x .+ b)
    output = output .* γ

    return output
end 



input = rand(Float32, (8, 32))
layer = DenseBatchEnsemble(8, 16, 1, 4)
# output = layer(input) 
print(layer)