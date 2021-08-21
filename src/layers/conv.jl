using Flux 
using Flux:@functor, glorot_uniform, create_bias, batch

struct ConvBatchEnsemble{L,F,M,B}
    layer::L 
    alpha::M
    gamma::M
    ensemble_bias::B
    ensemble_act::F
    rank::Integer
    function ConvBatchEnsemble(layer::L, alpha::M, gamma::M, 
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

function ConvBatchEnsemble(k::NTuple{N,Integer},
                            ch::Pair{<:Integer,<:Integer},
                            rank::Integer, 
                            ensemble_size::Integer,
                            σ=identity;
                            init=glorot_uniform,
                            alpha_init=glorot_normal, 
                            gamma_normal=glorot_normal, 
                            stride=1, 
                            pad=0, dilation=1,
                            groups=1, bias=true) where N
                            
    layer = Flux.Conv(k, ch, σ;
                    stride=stride, pad=pad, 
                    dilation=dilation, 
                    groups=groups, 
                    bias=bias)
    in_dim = ch[1] 
    out_dim = ch[2] 
    if rank >= 1 
        alpha_shape = (in_dim, ensemble_size)
        gamma_shape = (out_dim, ensemble_size) 
    else
        error("Rank must be >= 1.")
    end 
    alpha = init(alpha_shape) 
    gamma = init(gamma_shape)

    return ConvBatchEnsemble(layer, alpha, gamma, bias, σ, rank)
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
    output = layer(perturbed_x)  .* gamma
    output = e_σ.(output .+ e_b) 

    return output
end