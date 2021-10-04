"""
    DenseBE(in, out, rank, 
            ensemble_size, 
            σ=identity; 
            bias=true,
            init=glorot_normal, 
            alpha_init=glorot_normal, 
            gamma_init=glorot_normal)
    DenseBE(layer, alpha, gamma, ensemble_bias, ensemble_act, rank)

Creates a dense BatchEnsemble layer. Batch ensemble is a memory efficient alternative 
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
- `in::Integer`: Input dimension of features 
- `out::Integer`: Output dimension of features 
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
struct DenseBE{L,F,M,B}
    layer::L
    alpha::M
    gamma::M
    ensemble_bias::B
    ensemble_act::F
    rank
end

function DenseBE(
    layer,
    alpha,
    gamma,
    ensemble_bias = true,
    ensemble_act = identity,
    rank = 1,
)
    ensemble_bias = create_bias(gamma, ensemble_bias, size(gamma)[1], size(gamma)[2])
    DenseBE(layer, alpha, gamma, ensemble_bias, ensemble_act, rank)
end

function DenseBE(
    in::Integer,
    out::Integer,
    rank::Integer,
    ensemble_size::Integer,
    σ = identity;
    init = glorot_normal,
    alpha_init = glorot_normal,
    gamma_init = glorot_normal,
    bias = true,
    ensemble_bias = true,
    ensemble_act = identity,
)

    layer = Flux.Dense(in, out, σ; init = init, bias = bias)
    if rank >= 1
        alpha_shape = (in, ensemble_size, rank)
        gamma_shape = (out, ensemble_size, rank)
    else
        error("Rank must be >= 1.")
    end
    alpha = alpha_init(alpha_shape)
    gamma = gamma_init(gamma_shape)
    ensemble_bias = create_bias(gamma, ensemble_bias, size(gamma)[1], size(gamma)[2])

    return DenseBE(layer, alpha, gamma, ensemble_bias, ensemble_act, rank)
end

@functor DenseBE

"""
The forward pass for a DenseBE layer. The input is initially perturbed 
using the first fast weight, then passed through the dense layer, and finall 
multiplied by the second fast weight.

# Arguments 
- `x`: Input tensors 
"""
function (be::DenseBE)(x)
    layer = be.layer
    alpha = be.alpha
    gamma = be.gamma
    e_b = be.ensemble_bias
    e_σ = be.ensemble_act
    rank = be.rank

    batch_size = size(x)[end]
    in_size = size(alpha)[1]
    out_size = size(gamma)[1]
    ensemble_size = size(alpha)[2]
    samples_per_model = batch_size ÷ ensemble_size

    # TODO: Do we really need repeat? Or can we use broadcast? 
    # Alpha, gamma shapes - [units, ensembles, rank]
    alpha = reshape(alpha, (in_size, ensemble_size * rank))
    gamma = reshape(gamma, (out_size, ensemble_size * rank))
    # Repeat breaks on GPU when input dims > 2 
    alpha = repeat(alpha, samples_per_model)
    gamma = repeat(gamma, samples_per_model)
    # Reshape alpha, gamma to [units, batch_size, rank]
    alpha = reshape(alpha, (in_size, batch_size, rank))
    gamma = reshape(gamma, (out_size, batch_size, rank))
    # Reshape inputs to [units, batch_size, 1] for broadcasting
    x = Flux.unsqueeze(x, (ndims(x) + 1))
    # Perturb the inputs 
    perturbed_x = x .* alpha
    # Dense layer forward pass 
    outputs = layer(perturbed_x) .* gamma
    # Reduce the rank dimension through summing it up
    outputs = sum(outputs, dims = ndims(outputs))
    outputs = reshape(outputs, (out_size, samples_per_model, ensemble_size))
    # Reshape ensemble bias 
    e_b = Flux.unsqueeze(e_b, ndims(e_b))

    outputs = e_σ.(outputs .+ e_b)
    outputs = reshape(outputs, (out_size, batch_size))
    return outputs
end
