using DeepUncertainty
using Test
using Flux
using Flux.CUDA
using Flux: cpu, gpu
using DistributionsAD

@testset "Layers" begin
    # MC layers 
    include("./layers/mclayers.jl")
    # Batch ensembe layers 
    include("./layers/batchensemble.jl")
    # Variational layers 
    include("./layers/variational.jl")
    # Variational Batch ensembe layers 
    include("./layers/bayesian_be.jl")
    # Spectral layers 
    include("./layers/spectrallayers.jl")
end

@testset "Bayesian utils" begin
    include("bayesian.jl")
end

@testset "CUDA" begin
    if CUDA.functional()
        include("cuda/runtests.jl")
    else
        @warn "CUDA unavailable, not testing GPU support"
    end
end
