abstract type AbstractTrainableDist end

"""
    TrainableDistribution(shape;
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
- `prior_distribution`: The prior distribution function, defaults to a Multivariate Normal 
- `posterior_distribution`: The posterior distribution function, defaults to a Multivariate Normal 
- `mean_constraint`: Constraint on the mean of the distribution, defaults to identity
- `stddev_constraint`: Constrain on the stddev of the distrbution, defaults to softplus 
- `complexity_weight`: The regularization constant for KL divergence

# Arguments 
- `shape::Tuple`: The shape of the sample to returned from the distribution 
- `prior`: The prior distribution, defaults to Multivariate Normal 
- `posterior`: The posterior distribution, defaults to Multivariate Normal 
- `mean_constraint`: Constaint on the mean, defaults to identity 
- `stddev_constraint`: Constraint on stddev, defaults to softplus 
- `complexity_weight`: Regularization constant for KL divergence 
"""
struct TrainableDistribution{M,S,N,MF,SF,F,PD,POSD} <: AbstractTrainableDist
    mean::M
    stddev::M
    sample::S
    shape::NTuple{N,Integer}
    complexity_weight::F
    mean_constraint::MF
    stddev_constraint::SF
    prior_distribution::PD
    posterior_distribution::POSD
end

function TrainableDistribution(
    shape;
    complexity_weight = 1e-5,
    mean_init = glorot_normal,
    stddev_init = glorot_normal,
    mean_constraint = identity,
    stddev_constraint = softplus,
    prior_distribution = DistributionsAD.TuringMvNormal,
    posterior_distribution = DistributionsAD.TuringMvNormal,
)
    # Create the mean and stddev 
    total_params = prod(shape)
    mean = gpu(mean_init(total_params))
    stddev = gpu(stddev_init(total_params))
    sample = gpu(zeros(total_params))
    return TrainableDistribution(
        mean,
        stddev,
        sample,
        shape,
        complexity_weight,
        mean_constraint,
        stddev_constraint,
        prior_distribution,
        posterior_distribution,
    )
end

# Don't backprop through the sample stored for loss calc 
@functor TrainableDistribution (mean, stddev)

function (td::TrainableDistribution)()
    mean = td.mean_constraint.(td.mean)
    stddev = td.stddev_constraint.(td.stddev)
    dist = td.posterior_distribution(mean, stddev)
    # Sample from the dist 
    # Put it on zygote.ignore to supress mutation errors 
    sample = rand(dist)
    # Ignore the mutation array error while backprop
    Flux.Zygote.@ignore copyto!(td.sample, sample)
    sample = gpu(sample)
    return reshape(sample, td.shape)
end
