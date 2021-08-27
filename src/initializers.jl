using Flux
using Random, Distributions
using Flux: @functor, glorot_normal, glorot_uniform

"""
    TrainableGlorotNormal(shape, mean_initializer, stddev_initializer)
    TrainableGlorotNormal(mean, stddev)
A trainable layer which samples from a glorot normal distribution when called. 
During each forward pass, a random noise tensor is sampled from a normal distribution. 
The noise is multipltied but the log of exp of stddev and added to mean. This is the 
famous reparameterization trick. 

# Fields 
- `mean`: The mean of the distribution 
- `stddev`: The standard deviation of the RVs 
"""
struct TrainableGlorotNormal{M}
    mean::M
    stddev::M
    shape::Tuple
    function TrainableGlorotNormal(mean::M, stddev::M, shape::Tuple) where {M}
        new{M}(mean, stddev, shape)
    end
end

function TrainableGlorotNormal(
    shape::NTuple{N,Integer};
    mean_initializer = glorot_normal,
    stddev_initializer = glorot_normal,
) where {N}
    # total_weights = prod(shape)
    mean = mean_initializer(shape)
    stddev = stddev_initializer(shape)
    return TrainableGlorotNormal(mean, stddev, shape)
end

@functor TrainableGlorotNormal

function (l::TrainableGlorotNormal)()
    # TODO: compare this implementation with 
    # tfp one and see if it's mathematically equivalent 
    # sample random unit from gaussian dist 
    mean, stddev, shape = l.mean, l.stddev, l.shape
    # TODO: Get this to run on GPU 
    # dist = MvNormal(mean, stddev)
    # sample = rand(dist, 1)
    sigma = log.(1 .+ exp.(stddev))
    epsilon = rand(Normal(), shape)
    sample = mean .+ (sigma .* epsilon)
    # sample = reshape(sample, shape)
    return sample
end
