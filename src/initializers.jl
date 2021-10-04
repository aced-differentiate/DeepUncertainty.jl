abstract type AbstractTrainableDist end

"""
    TrainableMvNormal(shape;
                    prior=DistributionsAD.TuringMvNormal, 
                    posterior=DistributionsAD.TuringMvNormal, 
                    complexity_weight=1e-5)
    TrainableMvNormal(mean, stddev, sample, shape)

A distribution with trainable parameters. The mean and stddev are trainable parameters.
The prior and posterior are by default multivariate norml distribution. 

# Fields 
- `mean`: Trainable mean vector of the distribution 
- `stddev`: Trainable standard deviation vector of the distibution 
- `sample`: The latest sample from the distribution, used in calculating log likelhoods
- `shape::Tuple`: The shape of the sample to be returned

# Arguments 
- `shape::Tuple`: The shape of the sample to returned from the distribution 
- `init`: glorot_normal; to initialize the mean and stddev trainable params 
"""
struct TrainableMvNormal{M,S,N,D} <: AbstractTrainableDist
    mean::M
    stddev::M
    sample::S
    shape::NTuple{N,Integer}
    device::D
end

function TrainableMvNormal(shape; init = glorot_normal, device = cpu)
    # Create the mean and stddev 
    total_params = prod(shape)
    mean = device(init(total_params))
    stddev = device(init(total_params))
    sample = device(zeros(total_params))
    return TrainableMvNormal(mean, stddev, sample, shape, device)
end

# Don't backprop through the sample stored for loss calc 
@functor TrainableMvNormal (mean, stddev)

function (tmv::TrainableMvNormal)()
    dist = DistributionsAD.TuringMvNormal(tmv.mean, abs.(tmv.stddev))
    # Sample from the dist 
    # Put it on zygote.ignore to supress mutation errors 
    sample = tmv.device(rand(dist))
    # Ignore the mutation array error while backprop
    Flux.Zygote.@ignore copyto!(tmv.sample, sample)
    return reshape(sample, tmv.shape)
end
