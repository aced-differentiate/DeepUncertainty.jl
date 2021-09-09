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
mutable struct TrainableDistribution{M,S,N,MF,SF,F} <: AbstractTrainableDist
    mean::M 
    stddev::M 
    sample::S
    shape::NTuple{N,Integer}
    mean_constraint::MF 
    stddev_constraint::SF 
    complexity_weight::F
end 

function TrainableDistribution(shape; 
                                mean_init=glorot_normal, 
                                stddev_init=glorot_normal, 
                                complexity_weight=1e-5)
    # Create the mean and stddev 
    total_params = prod(shape) 
    mean = mean_init(total_params)
    stddev = stddev_init(total_params)
    sample = zeros(total_params)
    mean_constraint = identity 
    stddev_constraint = softplus
    return TrainableDistribution(mean, stddev, sample, shape,
                            mean_constraint, 
                            stddev_constraint, 
                            complexity_weight) 
end 

# Don't backprop through the sample stored for loss calc 
@functor TrainableDistribution (mean, stddev)

function (tn::TrainableDistribution)()
    mean = tn.mean_constraint.(tn.mean) 
    stddev = tn.stddev_constraint.(tn.stddev) 
    dist = DistributionsAD.TuringMvNormal(mean, stddev)
    # Sample from the dist 
    sample = gpu(rand(dist))
    # Assigning sample to struct moves it to CPU, even though its a CuArray 
    tn.sample = sample 
    return reshape(sample, tn.shape) 
end
