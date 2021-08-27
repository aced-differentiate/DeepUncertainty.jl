using Flux
using Random, Distributions
using Flux: @functor, glorot_normal, glorot_uniform

"""
    TrainableGlorotNormal(shape, loc_initializer, scale_initializer)
    TrainableGlorotNormal(loc, scale)
A trainable layer which samples from a glorot normal distribution when called. 
During each forward pass, a random noise tensor is sampled from a normal distribution. 
The noise is multipltied but the log of exp of scale and added to loc. This is the 
famous reparameterization trick. 

# Fields 
- `loc`: The loc of the distribution 
- `scale`: The standard deviation of the RVs 
"""
struct TrainableGlorotNormal{M}
    loc::M
    scale::M
    function TrainableGlorotNormal(loc::M, scale::M) where {M}
        new{M}(loc, scale)
    end
end

function TrainableGlorotNormal(
    shape::NTuple{N,Integer};
    loc_initializer = glorot_normal,
    scale_initializer = glorot_normal,
) where {N}
    loc = loc_initializer(shape)
    scale = scale_initializer(shape)
    return TrainableGlorotNormal(loc, scale)
end

@functor TrainableGlorotNormal

function (l::TrainableGlorotNormal)()
    # TODO: compare this implementation with 
    # tfp one and see if it's mathematically equivalent 
    # sample random unit from gaussian dist 
    loc, scale = l.loc, l.scale
    epsilon = rand(Normal(), size(loc))
    sigma = log.(1 .+ exp.(scale))
    sample = loc .+ (sigma .* epsilon)
    return sample
end
