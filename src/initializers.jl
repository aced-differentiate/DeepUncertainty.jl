abstract type AbstractTrainableDist end

"""
    TrainableMvNormal(shape;
                    prior=DistributionsAD.TuringMvNormal, 
                    posterior=DistributionsAD.TuringMvNormal, 
                    complexity_weight=1e-5)
A distribution with trainable parameters. The mean and stddev are trainable parameters.
The prior and posterior are by default multivariate norml distribution. 

# Fields 
- `mean`: Trainable mean vector of the distribution 
- `stddev`: Trainable standard deviation vector of the distibution 
- `sample`: The latest sample from the distribution, used in calculating log likelhoods
- `shape::Tuple`: The shape of the sample to be returned
- `complexity_weight`: The regularization constant for KL divergence
- `mean_constraint`: Constraint on the mean, defaults to identity 
- `stddev_constraint`: Constraint on stddev, defaults to softplus 

# Arguments 
- `shape::Tuple`: The shape of the sample to returned from the distribution 
- `prior`: The prior distribution, defaults to Multivariate Normal 
- `posterior`: The posterior distribution, defaults to Multivariate Normal 
- `complexity_weight`: Regularization constant for KL divergence 
"""
struct TrainableMvNormal{M,S,N,F,MF,SF} <: AbstractTrainableDist
    mean::M
    stddev::M
    sample::S
    shape::NTuple{N,Integer}
    complexity_weight::F
    mean_constraint::MF
    stddev_constraint::SF
end

function TrainableMvNormal(
    shape;
    complexity_weight = 1e-4,
    init = glorot_normal,
    mean_constraint = identity,
    stddev_constraint = softplus,
)
    # Create the mean and stddev 
    total_params = prod(shape)
    mean = gpu(init(total_params))
    stddev = gpu(init(total_params))
    sample = gpu(zeros(total_params))
    return TrainableMvNormal(
        mean,
        stddev,
        sample,
        shape,
        complexity_weight,
        mean_constraint,
        stddev_constraint,
    )
end

# Don't backprop through the sample stored for loss calc 
@functor TrainableMvNormal (mean, stddev)

function (tmv::TrainableMvNormal)()
    mean = tmv.mean_constraint.(tmv.mean)
    stddev = tmv.stddev_constraint.(tmv.stddev)
    dist = DistributionsAD.TuringMvNormal(mean, stddev)
    # Sample from the dist 
    # Put it on zygote.ignore to supress mutation errors 
    sample = rand(dist)
    # Ignore the mutation array error while backprop
    Flux.Zygote.@ignore copyto!(tmv.sample, sample)
    sample = gpu(sample)
    return reshape(sample, tmv.shape)
end
