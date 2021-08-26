using Flux 
using Random 
using Flux:@functor, glorot_uniform, create_bias
include("../initializers.jl")

struct MCDense{L,F}
    layer::L 
    dropout_rate::F
    function MCDense(layer::L, dropout_rate::F) where {L,F}
        new{typeof(layer),typeof(dropout_rate)}(layer, dropout_rate)
    end 
end 

function MCDense(in::Integer,
                out::Integer, 
                dropout_rate::AbstractFloat, 
                σ=identity;
                init=glorot_uniform, 
                bias=true)
    layer = Flux.Dense(in, out; init=init, bias=bias)
    return MCDense(layer, dropout_rate)
end 

@functor MCDense 

function (a::MCDense)(x::AbstractVecOrMat; dropout=true)
    output = a.layer(x) 
    output = Flux.dropout(output, a.dropout_rate;active=dropout)
    return output
end 

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
                                                    ensemble_act, rank)
    end
end
  
function DenseBatchEnsemble(in::Integer, out::Integer, 
                            rank::Integer, ensemble_size::Integer, 
                            σ=identity;
                            init=glorot_uniform,
                            alpha_init=glorot_uniform, 
                            gamma_init=glorot_uniform, 
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
    alpha = alpha_init(alpha_shape) 
    gamma = gamma_init(gamma_shape)
  
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
        alpha = repeat(alpha, samples_per_model)
        gamma = repeat(gamma, samples_per_model)
        # Reshape alpha, gamma to [units, batch_size, rank]
        alpha = reshape(alpha, (in_size, batch_size, rank))
        gamma = reshape(gamma, (out_size, batch_size, rank))
        # Reshape inputs to [units, batch_size, 1] for broadcasting
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

struct DenseRank1BatchEnsemble{L,I,F,B}
    layer::L 
    alpha_sampler::I
    gamma_sampler::I 
    ensemble_bias::B
    ensemble_act::F
    rank::Integer
    function DenseRank1BatchEnsemble(layer::L,
                                    alpha_sampler::I,
                                    gamma_sampler::I, 
                                    ensemble_bias=true, 
                                    ensemble_act::F=identity, 
                                    rank::Integer=1) where {F,L,I}
        gamma_shape = size(gamma_sampler.mean)
        ensemble_bias = create_bias(gamma_sampler.mean, ensemble_bias, gamma_shape[1], gamma_shape[2])
        new{typeof(layer),typeof(alpha_sampler),F,typeof(ensemble_bias)}(layer, 
                                                                        alpha_sampler, 
                                                                        gamma_sampler, 
                                                                        ensemble_bias, 
                                                                        ensemble_act, rank)
    end
end
  
function DenseRank1BatchEnsemble(in::Integer, out::Integer, 
                            rank::Integer, ensemble_size::Integer, 
                            σ=identity;
                            init=glorot_uniform,
                            alpha_init=glorot_uniform, 
                            gamma_init=glorot_uniform, 
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
    # Initialize alpha and gamma samplers 
    alpha_sampler = TrainableGlorotNormal(alpha_shape, 
                                        mean_initializer=alpha_init, 
                                        stddev_initializer=alpha_init)
    gamma_sampler = TrainableGlorotNormal(gamma_shape, 
                                        mean_initializer=gamma_init, 
                                        stddev_initializer=gamma_init)
  
    return DenseRank1BatchEnsemble(layer, alpha_sampler, gamma_sampler, bias, σ, rank)
end
  
@functor DenseRank1BatchEnsemble
  
function (a::DenseRank1BatchEnsemble)(x::AbstractVecOrMat)
    layer = a.layer 
    alpha = a.alpha_sampler()
    gamma = a.gamma_sampler()
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
        alpha = repeat(alpha, samples_per_model)
        gamma = repeat(gamma, samples_per_model)
        # Reshape alpha, gamma to [units, batch_size, rank]
        alpha = reshape(alpha, (in_size, batch_size, rank))
        gamma = reshape(gamma, (out_size, batch_size, rank))
        # Reshape inputs to [units, batch_size, 1] for broadcasting
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