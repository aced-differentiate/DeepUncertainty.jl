using Flux, Test, CUDA

@info "Testing GPU Support"
CUDA.allowscalar(false)

include("bayesian.jl")
include("layers/mclayers.jl")
include("layers/batchensemble.jl")
include("layers/variational.jl")
# include("layers/bayesianBE.jl")
