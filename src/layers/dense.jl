using Flux
using Random
using Flux: @functor, glorot_uniform, create_bias
include("../initializers.jl")

"""
    MCDense(in, out, dropout_rate, σ=identity; bias=true, init=glorot_uniform)
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
- `dropout_rate::AbstractFloat`: Dropout rate 

# Arguments 
- `in::Integer`: Input dimension of features 
- `out::Integer`: Output dimension of features 
- `dropout_rate::AbstractFloat`: Dropout rate 
- `σ::F=identity`: Activation function, defaults to identity
- `init=glorot_normal`: Initialization function, defaults to glorot_normal 
"""
struct MCDense{L,F}
    layer::L
    dropout_rate::F
    function MCDense(layer::L, dropout_rate::F) where {L,F}
        new{typeof(layer),typeof(dropout_rate)}(layer, dropout_rate)
    end
end

function MCDense(
    in::Integer,
    out::Integer,
    dropout_rate::AbstractFloat,
    σ = identity;
    init = glorot_normal,
    bias = true,
)

    layer = Flux.Dense(in, out, σ; init = init, bias = bias)
    return MCDense(layer, dropout_rate)
end

@functor MCDense

"""
The forward pass of a MCDense layer: Passes the input through the 
usual dense layer first and then through a dropout layer. 

# Arguments 
- `x::AbstractVecOrMat`: Input tensors 
- `dropout::Bool=true`: Toggle to control dropout, it's preferred to keep 
dropout always on, but just in case if it's needed. 
"""
function (a::MCDense)(x::AbstractVecOrMat; dropout = true)
    output = a.layer(x)
    output = Flux.dropout(output, a.dropout_rate; active = dropout)
    return output
end

"""
    DenseBatchEnsemble(in, out, rank, 
                        ensemble_size, 
                        σ=identity; 
                        bias=true,
                        init=glorot_normal, 
                        alpha_init=glorot_normal, 
                        gamma_init=glorot_normal)
    DenseBatchEnsemble(layer, alpha, gamma, ensemble_bias, ensemble_act, rank)

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
struct DenseBatchEnsemble{L,F,M,B}
    layer::L
    alpha::M
    gamma::M
    ensemble_bias::B
    ensemble_act::F
    rank::Integer
    function DenseBatchEnsemble(
        layer::L,
        alpha::M,
        gamma::M,
        ensemble_bias = true,
        ensemble_act::F = identity,
        rank::Integer = 1,
    ) where {M,F,L}
        ensemble_bias = create_bias(gamma, ensemble_bias, size(gamma)[1], size(gamma)[2])
        new{typeof(layer),F,M,typeof(ensemble_bias)}(
            layer,
            alpha,
            gamma,
            ensemble_bias,
            ensemble_act,
            rank,
        )
    end
end

function DenseBatchEnsemble(
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

    return DenseBatchEnsemble(layer, alpha, gamma, ensemble_bias, ensemble_act, rank)
end

@functor DenseBatchEnsemble

"""
The forward pass for a DenseBatchEnsemble layer. The input is initially perturbed 
using the first fast weight, then passed through the dense layer, and finall 
multiplied by the second fast weight.

# Arguments 
- `x::AbstractVecOrMat`: Input tensors 
"""
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
        perturbed_x = x .* alpha
        # Dense layer forward pass 
        outputs = layer(perturbed_x) .* gamma
        # Reduce the rank dimension through summing it up
        outputs = sum(outputs, dims = 3)
        outputs = reshape(outputs, (out_size, samples_per_model, ensemble_size))
    else
        # Reshape the inputs, alpha and gamma
        x = reshape(x, (in_size, samples_per_model, ensemble_size))
        alpha = reshape(alpha, (in_size, 1, ensemble_size))
        gamma = reshape(gamma, (out_size, 1, ensemble_size))
        # Perturb the inputs 
        perturbed_x = x .* alpha
        # Dense layer forward pass 
        outputs = layer(perturbed_x) .* gamma
    end
    # Reshape ensemble bias 
    e_b = reshape(e_b, (out_size, 1, ensemble_size))
    outputs = e_σ.(outputs .+ e_b)
    outputs = reshape(outputs, (out_size, batch_size))
    return outputs
end

"""
    DenseBayesianBatchEnsemble(in, out, rank, 
                                ensemble_size, 
                                σ=identity; 
                                bias=true,
                                init=glorot_normal, 
                                alpha_init=glorot_normal, 
                                gamma_init=glorot_normal)
    DenseBayesianBatchEnsemble(layer, alpha_sampler, gamma_sampler,
                                ensemble_bias, ensemble_act, rank)

Creates a bayesian dense BatchEnsemble layer. 
Batch ensemble is a memory efficient alternative for deep ensembles. In deep ensembles, 
if the ensemble size is N, N different models are trained, making the time and memory 
complexity O(N * complexity of one network). 
BatchEnsemble generates weight matrices for each member in the ensemble using a 
couple of rank 1 vectors R (alpha), S (gamma), RS' and multiplying the result with 
weight matrix W element wise. We also call R and S as fast weights. In the bayesian 
version of batch ensemble, instead of having point estimates of the fast weights, we 
sample them form a trainable parameterized distribution. 

Reference - https://arxiv.org/abs/2005.07186

During both training and testing, we repeat the samples along the batch dimension 
N times, where N is the ensemble_size. For example, if each mini batch has 10 samples 
and our ensemble size is 4, then the actual input to the layer has 40 samples. 
The output of the layer has 40 samples as well, and each 10 samples can be considered 
as the output of an esnemble member. 

# Fields 
- `layer`: The dense layer which transforms the pertubed input to output 
- `alpha_sampler`: Sampler for the first Fast weight of size (in_dim, ensemble_size)
- `gamma_sampler`: Sampler for the second Fast weight of size (out_dim, ensemble_size)
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
                            defaults to TrainableGlorotNormal 
- `gamma_init=glorot_normal`: Initialization function for the gamma fast weight, 
                            defaults to TrainableGlorotNormal 
- `bias::Bool=true`: Toggle the usage of bias in the dense layer 
- `ensemble_bias::Bool=true`: Toggle the usage of ensemble bias 
- `ensemble_act::F=identity`: Activation function for enseble outputs 
"""
struct DenseBayesianBatchEnsemble{L,I,F,B}
    layer::L
    alpha_sampler::I
    gamma_sampler::I
    ensemble_bias::B
    ensemble_act::F
    rank::Integer
    function DenseBayesianBatchEnsemble(
        layer::L,
        alpha_sampler::I,
        gamma_sampler::I,
        ensemble_bias = true,
        ensemble_act::F = identity,
        rank::Integer = 1,
    ) where {F,L,I}
        gamma_shape = size(gamma_sampler.mean)
        ensemble_bias =
            create_bias(gamma_sampler.mean, ensemble_bias, gamma_shape[1], gamma_shape[2])
        new{typeof(layer),typeof(alpha_sampler),F,typeof(ensemble_bias)}(
            layer,
            alpha_sampler,
            gamma_sampler,
            ensemble_bias,
            ensemble_act,
            rank,
        )
    end
end

function DenseBayesianBatchEnsemble(
    in::Integer,
    out::Integer,
    rank::Integer,
    ensemble_size::Integer,
    σ = identity;
    init = glorot_normal,
    alpha_init = TrainableGlorotNormal,
    gamma_init = TrainableGlorotNormal,
    bias = true,
    ensemble_bias = true,
    ensemble_act = identity,
)

    layer = Flux.Dense(in, out, σ; init = init, bias = bias)
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
    alpha_sampler = TrainableGlorotNormal(
        alpha_shape,
        mean_initializer = alpha_init,
        stddev_initializer = alpha_init,
    )
    gamma_sampler = TrainableGlorotNormal(
        gamma_shape,
        mean_initializer = gamma_init,
        stddev_initializer = gamma_init,
    )

    return DenseBayesianBatchEnsemble(
        layer,
        alpha_sampler,
        gamma_sampler,
        ensemble_bias,
        ensemble_act,
        rank,
    )
end

@functor DenseBayesianBatchEnsemble

"""
The forward pass for a DenseBayesianBatchEnsemble layer. 
The fast weights are sampled from trainable distributions. The input is 
initially perturbed using the first fast weight, then passed through the 
dense layer, and finall multiplied by the second fast weight.

# Arguments 
- `x::AbstractVecOrMat`: Input tensors 
"""
function (a::DenseBayesianBatchEnsemble)(x::AbstractVecOrMat)
    layer = a.layer
    # Sample the fast weights from trainable distributions  
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
        perturbed_x = x .* alpha
        # Dense layer forward pass 
        outputs = layer(perturbed_x) .* gamma
        # Reduce the rank dimension through summing it up
        outputs = sum(outputs, dims = 3)
        outputs = reshape(outputs, (out_size, samples_per_model, ensemble_size))
    else
        # Reshape the inputs, alpha and gamma
        x = reshape(x, (in_size, samples_per_model, ensemble_size))
        alpha = reshape(alpha, (in_size, 1, ensemble_size))
        gamma = reshape(gamma, (out_size, 1, ensemble_size))
        # Perturb the inputs 
        perturbed_x = x .* alpha
        # Dense layer forward pass 
        outputs = layer(perturbed_x) .* gamma
    end
    # Reshape ensemble bias 
    e_b = reshape(e_b, (out_size, 1, ensemble_size))
    outputs = e_σ.(outputs .+ e_b)
    outputs = reshape(outputs, (out_size, batch_size))
    return outputs
end
