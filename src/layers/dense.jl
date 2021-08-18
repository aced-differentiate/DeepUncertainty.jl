using Flux 
using Flux:@functor, glorot_uniform, create_bias

struct DenseBatchEnsemble{L,F,M,B}
    layer::L 
    alpha::M
    gamma::M
    ensemble_bias::B
    ensemble_act::F
    rank::Integer
    function DenseBatchEnsemble(layer::L, alpha::M, gamma::M, 
                                ensemble_bias=true, 
                                ensemble_act::F=identity, 
                                rank::Integer=1) where {M,F,L}
        ensemble_bias = create_bias(gamma, ensemble_bias, size(gamma)[1], size(gamma)[2])
        new{typeof(layer),F,M,typeof(ensemble_bias)}(layer, alpha, gamma, 
                                                    ensemble_bias, 
                                                    ensemble_act, 
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
        alpha_shape = (in, ensemble_size, rank)
        gamma_shape = (out, ensemble_size, rank)
    elseif rank == 1 
        alpha_shape = (in, ensemble_size)
        gamma_shape = (out, ensemble_size) 
    else
        error("Rank must be >= 1.")
    end 
    alpha = init(alpha_shape) 
    gamma = init(gamma_shape)
  
    return DenseBatchEnsemble(layer, alpha, gamma, bias, σ, rank)
end
  
@functor DenseBatchEnsemble
  
function (a::DenseBatchEnsemble)(x::AbstractVecOrMat)
    layer = a.layer 
    alpha = a.alpha
    gamma = a.gamma
    e_b = a.ensemble_bias 
    e_σ = a.ensemble_act
    rank = a.rank

    batch_size = size(x)[end]
    in_size = size(alpha)[1]
    out_size = size(gamma)[1]
    ensemble_size = size(alpha)[2]
    samples_per_model = batch_size ÷ ensemble_size 

    # TODO: Merge these implementations 
    if rank > 1 
        # Alpha, gamma shapes - [units, ensembles, rank]
        alpha = repeat(alpha, outer=[samples_per_model, 1, 1])
        gamma = repeat(gamma, outer=[samples_per_model, 1, 1])
        # Reshape alpha, gamma to [units, batch_size, rank]
        alpha = reshape(alpha, (in_size, batch_size, rank))
        gamma = reshape(gamma, (out_size, batch_size, rank))
        x = reshape(x, (in_size, batch_size, 1))
        # Perturb the inputs 
        perturbed_x = x .* alpha;
        # Dense layer forward pass 
        outputs = layer(perturbed_x) .* gamma
        # Reduce the rank dimension through summing it up
        outputs = sum(outputs, dims=3)
        outputs = reshape(outputs, (out_size, samples_per_model, ensemble_size))
    else
        # Reshape the inputs, alpha and gamma
        x = reshape(x, (in_size, samples_per_model, ensemble_size))
        alpha = reshape(alpha, (in_size, 1, ensemble_size))
        gamma = reshape(gamma, (out_size, 1, ensemble_size))
        # Perturb the inputs 
        perturbed_x = x .* alpha;
        # Dense layer forward pass 
        outputs = layer(perturbed_x)  .* gamma
    end 
    # Reshape ensemble bias 
    e_b = reshape(e_b, (out_size, 1, ensemble_size))
    outputs = e_σ.(outputs .+ e_b)
    outputs = reshape(outputs, (out_size, batch_size))
    return outputs
end
  
(a::DenseBatchEnsemble)(x::AbstractArray) = 
    reshape(a(reshape(x, size(x, 1), :)), :, size(x)[2:end]...)
  
function Base.show(io::IO, l::DenseBatchEnsemble)
    print(io, "DenseBatchEnsemble(", size(l.layer.weight, 2), ", ", size(l.layer.weight, 1))
    l.layer.σ == identity || print(io, ", ", l.layer.σ)
    l.layer.bias == Zeros() && print(io, "; bias=false")
    print(io, ")")
end