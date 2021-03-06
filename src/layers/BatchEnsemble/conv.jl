"""
    ConvBE(filter, in => out, rank, 
            ensemble_size, σ = identity;
            stride = 1, pad = 0, dilation = 1, 
            groups = 1, [bias, weight, init])
    ConvBE(layer, alpha, gamma, ensemble_bias, ensemble_act, rank)

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
struct ConvBE{L,F,M,B}
    layer::L
    alpha::M
    gamma::M
    ensemble_bias::B
    ensemble_act::F
    rank
end

function ConvBE(
    layer,
    alpha,
    gamma,
    ensemble_bias = true,
    ensemble_act = identity,
    rank = 1,
)
    ensemble_bias = create_bias(gamma, ensemble_bias, size(gamma)[1], size(gamma)[2])
    ConvBE(layer, alpha, gamma, ensemble_bias, ensemble_act, rank)
end

function ConvBE(
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
    ensemble_bias = create_bias(gamma, ensemble_bias, out_dim, ensemble_size)

    return ConvBE(layer, alpha, gamma, ensemble_bias, ensemble_act, rank)
end

@functor ConvBE

function (be::ConvBE)(x)
    # Conv Batch Ensemble params 
    layer = be.layer
    alpha = be.alpha
    gamma = be.gamma
    e_b = be.ensemble_bias
    e_σ = be.ensemble_act

    batch_size = size(x)[end]
    in_size = size(alpha)[1]
    out_size = size(gamma)[1]
    ensemble_size = size(alpha)[2]
    samples_per_model = batch_size ÷ ensemble_size

    # TODO: Do we really need repeat? Or can we use broadcast?
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
