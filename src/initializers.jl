using Flux 
using Random, Distributions
using Flux:@functor, glorot_normal, glorot_uniform

struct TrainableGlorotNormal{M}
    mean::M
    stddev::M 
    function TrainableGlorotNormal(mean::M, stddev::M) where {M}
        new{M}(mean, stddev)
    end 
end 

function TrainableGlorotNormal(shape::NTuple{N,Integer};
                        mean_initializer=glorot_normal,
                        stddev_initializer=glorot_normal) where {N}
    mean = mean_initializer(shape)
    stddev = stddev_initializer(shape)
    return TrainableGlorotNormal(mean, stddev)
end 

@functor TrainableGlorotNormal 

function (l::TrainableGlorotNormal)()
    # TODO: compare this implementation with 
    # tfp one and see if it's mathematically equivalent 
    # sample random unit from gaussian dist 
    mean, stddev = l.mean, l.stddev
    epsilon = rand(Normal(), size(mean))
    sigma = log.(1 .+ exp.(stddev))
    sample = mean .+ (sigma .* epsilon)
    return sample 
end 